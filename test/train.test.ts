import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, writeFile, rm, rename, readdir } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { train, evaluateBundle } from '../src/train.js';
import { loadClassifier } from '../src/index.js';
import { trainHead, predictHead } from '../src/linear.js';
import { metrics } from '../src/metrics.js';

const fixture = { task: 'examples/task.json', data: 'examples/training.jsonl', validation: 'examples/validation.jsonl', test: 'examples/test.jsonl', backend: 'tfidf' as const };

test('train/export/move/reload runs entirely offline with separate test metrics and advisory recommendations', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'jev-train-'));
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => { throw new Error('Unexpected network access'); };
  try {
    const result = await train({ ...fixture, out: join(dir, 'model') });
    assert.equal(result.report.test?.humanExamples, 9);
    assert.ok(result.report.test!.humanAccuracy! > 0.8);
    assert.equal(result.report.validation.humanAccuracy, null);
    assert.equal(result.report.dataset.train.count, 36);
    assert.ok(result.report.training.bestEpoch > 0);
    await rename(join(dir, 'model'), join(dir, 'moved'));
    const classifier = await loadClassifier(join(dir, 'moved'));
    try {
      const prediction = await classifier.predict('The subscription invoice shows a duplicate charge.');
      assert.equal(prediction.answer.type, 'choice');
      if (prediction.answer.type === 'choice') assert.equal(prediction.answer.choice, 'billing');
      assert.deepEqual(Object.keys(prediction).sort(), ['answer', 'maxProbability']);
      assert.equal(classifier.metadata.thresholdRecommendation.threshold, null);
      const unknown = await classifier.predict('zyzzyva quokka gnu');
      assert.equal(unknown.answer.type, 'choice');
      assert.deepEqual(Object.keys(unknown).sort(), ['answer', 'maxProbability']);
      const result = await classifier.evaluate({ state: 'The package tracking is late' });
      assert.ok(result.answers.route);
      assert.deepEqual(Object.keys(result).sort(), ['answers', 'model']);
      assert.deepEqual(await classifier.predictBatch([]), []);
      await assert.rejects(classifier.predict('invoice '.repeat(5000)), /supports 4096/);
    } finally { await classifier.dispose(); }
    await assert.rejects(classifier.predict('hello'), /disposed/);
    const evaluated = await evaluateBundle(join(dir, 'moved'), 'examples/test.jsonl');
    assert.equal(evaluated.metrics.humanAccuracy, result.report.test!.humanAccuracy);
  } finally { globalThis.fetch = originalFetch; await rm(dir, { recursive: true, force: true }); }
});

test('soft-label loss learns uncertainty, not just the winning class', () => {
  const x = [{ indices: Uint32Array.of(0), values: Float32Array.of(1) }, { indices: Uint32Array.of(0), values: Float32Array.of(-1) }];
  const targets = [[0.7, 0.3], [0.2, 0.8]];
  const trained = trainHead(x, targets, 1, { vectors: x, targets }, { epochs: 300, learningRate: 0.05, l2: 0 });
  assert.ok(Math.abs(predictHead(trained.head, x[0])[0] - 0.7) < 0.015);
  assert.ok(Math.abs(predictHead(trained.head, x[1])[1] - 0.8) < 0.015);
});

test('vocabulary sees training only, and repeated runs have identical learned weights', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'jev-repeat-'));
  try {
    const validation = JSON.parse('[' + (await readFile(fixture.validation, 'utf8')).trim().split('\n').join(',') + ']');
    validation[0].state += ' SECRETVALIDATIONTOKEN';
    await writeFile(join(dir, 'validation.json'), JSON.stringify(validation));
    const options = { ...fixture, validation: join(dir, 'validation.json') };
    const a = await train({ ...options, out: join(dir, 'a') });
    const b = await train({ ...options, out: join(dir, 'b') });
    assert.deepEqual(a.manifest.head, b.manifest.head);
    assert.deepEqual(a.manifest.features, b.manifest.features);
    assert.equal(a.report.dataset.train.sha256, b.report.dataset.train.sha256);
    assert.equal(a.manifest.features.kind, 'tfidf');
    if (a.manifest.features.kind === 'tfidf') assert.ok(!a.manifest.features.vocabulary.some(v => v.includes('secretvalidationtoken')));
  } finally { await rm(dir, { recursive: true, force: true }); }
});

test('rejects validation/test leakage, invalid hyperparameters, and existing output directories', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'jev-errors-'));
  try {
    await assert.rejects(train({ ...fixture, out: join(dir, 'a'), validation: fixture.data }), /overlaps/);
    await assert.rejects(train({ ...fixture, out: join(dir, 'a'), test: fixture.validation }), /overlaps/);
    await assert.rejects(train({ ...fixture, out: join(dir, 'a'), epochs: -1 }), /epochs/);
    await assert.rejects(train({ ...fixture, out: join(dir, 'a'), learningRate: Infinity }), /learning-rate/);
    await train({ ...fixture, out: join(dir, 'a') });
    await assert.rejects(train({ ...fixture, out: join(dir, 'a') }), /already exists/);
    assert.deepEqual(await readdir(dir), ['a']);
  } finally { await rm(dir, { recursive: true, force: true }); }
});

test('detects damaged manifest and report before inference', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'jev-corrupt-'));
  try {
    const out = join(dir, 'model');
    await train({ ...fixture, out });
    const original = await readFile(join(out, 'model.json'), 'utf8');
    const corrupt = JSON.parse(original);
    corrupt.head.weights[0].pop();
    await writeFile(join(out, 'model.json'), JSON.stringify(corrupt));
    await assert.rejects(loadClassifier(out), /weight dimensions/);
    await writeFile(join(out, 'model.json'), original);
    await writeFile(join(out, 'report.json'), '{}');
    await assert.rejects(loadClassifier(out), /Checksum mismatch/);
  } finally { await rm(dir, { recursive: true, force: true }); }
});

test('boolean, Noul, and Score bundles round trip into their original answer shapes', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'jev-types-'));
  try {
    for (const type of ['boolean', 'noul', 'score'] as const) {
      const question = { type, instructions: 'Is the message positive?', ...(type === 'score' ? { criteria: ['negative', 'positive'] } : {}) };
      const task = join(dir, `${type}-task.json`), data = join(dir, `${type}-data.json`);
      await writeFile(task, JSON.stringify({ questions: { sentiment: question } }));
      await writeFile(data, JSON.stringify(Array.from({ length: 20 }, (_, i) => ({
        state: `${i < 10 ? 'happy excellent wonderful' : 'sad terrible awful'} message number ${i}`,
        label: type === 'score' ? Number(i < 10) : i < 10,
      }))));
      const out = join(dir, type);
      await train({ task, data, out, backend: 'tfidf' });
      const classifier = await loadClassifier(out);
      try {
        const result = await classifier.predict('happy wonderful excellent');
        assert.equal(result.answer.type, type);
        if (result.answer.type === 'boolean') assert.ok(result.answer.probability > 0.8);
        if (result.answer.type === 'noul') assert.ok(result.answer.noul > 0.8);
        if (result.answer.type === 'score') assert.ok(result.answer.score > 0.8);
      } finally { await classifier.dispose(); }
    }
  } finally { await rm(dir, { recursive: true, force: true }); }
});

test('reports teacher agreement separately from human correctness and null for zero coverage', () => {
  const examples = [{ state: 'x', text: 'x', target: [0, 1], teacher: [1, 0], humanLabel: 1 }];
  const result = metrics(examples, [[0.9, 0.1]], ['no', 'yes'], 0.99);
  assert.equal(result.teacherAgreement, 1);
  assert.equal(result.humanAccuracy, 0);
  assert.equal(result.operatingPoint.coverage, 0);
  assert.equal(result.operatingPoint.targetAgreement, null);
});

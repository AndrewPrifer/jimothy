import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { accuracyLowerBound, fitTemperature, hardLogLoss, selectAcceptance } from '../src/calibration.js';
import { argmax, assertDisjoint, parseDataset, splitDevelopment } from '../src/data.js';
import { softmax } from '../src/linear.js';
import { parseTask } from '../src/schema.js';
import { train, evaluateBundle } from '../src/train.js';
import { loadClassifier } from '../src/sdk.js';
import { readManifest, writeManifest } from '../src/artifact.js';

const taskJson = { questions: { intent: { type: 'choice', instructions: 'Pick the topic.', criteria: { billing: 'Payments', shipping: 'Delivery' } } } };
const task = parseTask(taskJson);

test('temperature recovers known underconfidence and overconfidence without changing decisions', () => {
  for (const multiplier of [0.25, 2]) {
    const logits = Array.from({ length: 100 }, () => [Math.log(9) * multiplier, 0]);
    const labels = Array.from({ length: 100 }, (_, i) => i < 90 ? 0 : 1);
    const fit = fitTemperature(logits, labels);
    assert.equal(fit.status, 'fitted');
    assert.ok(Math.abs(fit.temperature - multiplier) < 0.001);
    assert.ok(fit.lossAfter! < fit.lossBefore!);
    assert.ok(Math.abs(softmax(logits[0], fit.temperature)[0] - 0.9) < 0.001);
    assert.equal(argmax(softmax(logits[0], fit.temperature)), argmax(softmax(logits[0])));
  }
  assert.ok(Number.isFinite(hardLogLoss([[10_000, -10_000]], [1], 0.05)));
  assert.equal(fitTemperature([[1, 0]], [0]).temperature, 1);
  assert.equal(fitTemperature([[1, 0]], [0]).status, 'insufficient_data');
});

test('exact binomial lower bounds match analytic boundary cases and account for sample size', () => {
  assert.equal(accuracyLowerBound(0, 0), 0);
  assert.equal(accuracyLowerBound(0, 100), 0);
  assert.ok(Math.abs(accuracyLowerBound(100, 100) - 0.05 ** 0.01) < 1e-12);
  assert.ok(Math.abs(accuracyLowerBound(1, 100) - (1 - 0.95 ** 0.01)) < 1e-12);
  assert.ok(accuracyLowerBound(95, 100) < accuracyLowerBound(950, 1000));
  assert.ok(accuracyLowerBound(95, 100, 0.05 / 13) < accuracyLowerBound(95, 100));
  assert.throws(() => accuracyLowerBound(101, 100), /Invalid/);
});

test('acceptance maximizes coverage using supported thresholds and never treats a small perfect sample as proof', () => {
  const probabilities = Array.from({ length: 200 }, (_, i) => i < 170 ? [0.99, 0.01] : [0.6, 0.4]);
  const labels = probabilities.map((_, i) => i < 170 ? 0 : 1);
  const result = selectAcceptance(probabilities, labels, labels.map(() => true), 0.95);
  assert.equal(result.acceptance.status, 'ready');
  assert.equal(result.threshold, 0.7);
  const chosen = result.candidates.find(c => c.threshold === result.threshold)!;
  assert.equal(chosen.accepted, 170);
  assert.ok(chosen.lowerBound >= 0.95);
  assert.equal(selectAcceptance([[1, 0]], [0], [true], 0.95).acceptance.status, 'insufficient_data');
  assert.equal(selectAcceptance(probabilities, labels.map(() => 1), labels.map(() => true), 0.95).acceptance.status, 'target_not_met');
  assert.equal(selectAcceptance(probabilities, labels, labels.map(() => false), 0.95).acceptance.status, 'target_not_met');
  assert.equal(selectAcceptance(probabilities, labels, labels.map(() => true), 0.95, false).acceptance.status, 'insufficient_data');
});

test('development groups stay disjoint and split independently of source row order', () => {
  const examples = parseDataset(Array.from({ length: 120 }, (_, i) => ({ state: `row ${i}`, group: `group${Math.floor(i / 3)}`,
    label: i < 60 ? 'billing' : 'shipping' })), task).examples;
  const first = splitDevelopment(examples, 42), reversed = splitDevelopment([...examples].reverse(), 42);
  assert.deepEqual(first, reversed);
  assert.equal(first.tuning.length, 60);
  assert.equal(first.calibration.length, 30);
  assert.equal(first.acceptance.length, 30);
  assertDisjoint(first.tuning, first.calibration, 'Calibration');
  assertDisjoint(first.tuning, first.acceptance, 'Acceptance');
  assertDisjoint(first.calibration, first.acceptance, 'Acceptance');
});

test('automatic training selects and exports a recommendation without consulting test labels', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'jev-auto-'));
  const rows = (count: number, split: string, reverse = false) => Array.from({ length: count }, (_, i) => ({
    state: `${i % 2 ? 'parcel delivery shipping' : 'invoice payment billing'} ${split} example ${i}`,
    label: (i % 2 === 0) !== reverse ? 'billing' : 'shipping',
  }));
  try {
    const paths = { task: join(dir, 'task.json'), data: join(dir, 'train.json'), validation: join(dir, 'dev.json'), test: join(dir, 'test.json') };
    await writeFile(paths.task, JSON.stringify(taskJson));
    await writeFile(paths.data, JSON.stringify(rows(80, 'train')));
    await writeFile(paths.validation, JSON.stringify(rows(480, 'development')));
    await writeFile(paths.test, JSON.stringify(rows(20, 'test')));
    const options = { ...paths, backend: 'tfidf' as const, epochs: 40 };
    const a = await train({ ...options, out: join(dir, 'a') });
    assert.equal(a.manifest.version, 3);
    assert.equal(a.report.training.candidates.length, 5);
    assert.equal(a.manifest.calibration.status, 'fitted');
    assert.equal(a.manifest.thresholdRecommendation.status, 'ready');
    assert.equal(a.report.summary.accuracy, 1);
    assert.equal(a.report.summary.coverage, 1);
    await writeFile(paths.test, JSON.stringify(rows(20, 'test', true)));
    const b = await train({ ...options, out: join(dir, 'b') });
    assert.deepEqual(a.manifest.head, b.manifest.head);
    assert.deepEqual(a.manifest.calibration, b.manifest.calibration);
    assert.deepEqual(a.manifest.thresholdRecommendation, b.manifest.thresholdRecommendation);
    assert.equal('threshold' in a.manifest, false);
    assert.equal('acceptance' in a.manifest, false);
    assert.equal(b.report.summary.accuracy, 0);
    const model = await loadClassifier(join(dir, 'a'));
    try {
      const prediction = await model.predict('invoice billing');
      assert.equal('accepted' in prediction, false);
      assert.deepEqual(model.metadata.thresholdRecommendation, a.report.thresholdRecommendation);
      assert.ok(prediction.maxProbability >= model.metadata.thresholdRecommendation.threshold!);
      const metadata = model.metadata;
      metadata.thresholdRecommendation.threshold = 1;
      assert.notEqual(model.metadata.thresholdRecommendation.threshold, 1);
    } finally { await model.dispose(); }
    const fixed = await train({ ...options, l2: 0.001, out: join(dir, 'fixed') });
    assert.equal(fixed.report.training.candidates.length, 1);
  } finally { await rm(dir, { recursive: true, force: true }); }
});

test('SDK applies calibration and returns predictions when no threshold is recommended', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'jev-policy-'));
  try {
    await writeFile(join(dir, 'report.json'), '{}');
    const base = { format: 'jev-distill' as const, version: 3 as const, createdAt: new Date().toISOString(), task,
      preprocessing: 'canonical-json-v1' as const,
      features: { kind: 'tfidf' as const, vocabulary: ['w:invoice'], idf: [1], maxTokens: 4096 },
      head: { weights: [[Math.log(9)], [0]], bias: [0, 0] } };
    await writeManifest(dir, { ...base, calibration: { method: 'temperature', status: 'fitted', temperature: 2 },
      thresholdRecommendation: { threshold: 0.8, status: 'ready', targetAccuracy: 0.95 } });
    const calibrated = await loadClassifier(dir);
    try {
      const p = await calibrated.predict('invoice');
      assert.ok(Math.abs(p.maxProbability - 0.75) < 1e-8);
      assert.equal('accepted' in p, false);
    } finally { await calibrated.dispose(); }
    await writeManifest(dir, { ...base, head: { weights: [[10_000], [0]], bias: [0, 0] },
      calibration: { method: 'temperature', status: 'insufficient_data', temperature: 1 },
      thresholdRecommendation: { threshold: null, status: 'insufficient_data', targetAccuracy: 0.95 } });
    const disabled = await loadClassifier(dir);
    try {
      const p = await disabled.predict('invoice');
      assert.equal(p.maxProbability, 1);
      assert.equal('accepted' in p, false);
      assert.equal(disabled.metadata.thresholdRecommendation.threshold, null);
    } finally { await disabled.dispose(); }
    await writeFile(join(dir, 'test.json'), JSON.stringify([{ state: 'invoice', label: 'billing' }]));
    const evaluation = await evaluateBundle(dir, join(dir, 'test.json'));
    assert.equal(evaluation.summary.coverage, null);
    assert.equal(evaluation.metrics.operatingPoint, null);
    assert.ok(evaluation.metrics.coverageCurve.length > 0);
    const original = await readFile(join(dir, 'model.json'), 'utf8');
    const changedRecommendation = JSON.parse(original);
    changedRecommendation.thresholdRecommendation.targetAccuracy = 0.9;
    await writeFile(join(dir, 'model.json'), JSON.stringify(changedRecommendation));
    await assert.rejects(loadClassifier(dir), /checksum mismatch/);
    const corrupt = JSON.parse(original);
    corrupt.calibration.temperature = -1;
    await writeFile(join(dir, 'model.json'), JSON.stringify(corrupt));
    await assert.rejects(loadClassifier(dir), /calibration/);
  } finally { await rm(dir, { recursive: true, force: true }); }
});

test('v3 recommendation metadata is advisory, supports a zero cutoff, and is integrity checked', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'jev-recommendation-'));
  try {
    await writeFile(join(dir, 'report.json'), '{}');
    const base = { format: 'jev-distill' as const, version: 3 as const, createdAt: new Date().toISOString(), task,
      preprocessing: 'canonical-json-v1' as const,
      features: { kind: 'tfidf' as const, vocabulary: ['w:invoice'], idf: [1], maxTokens: 4096 },
      head: { weights: [[Math.log(9)], [0]], bias: [0, 0] },
      calibration: { method: 'temperature' as const, status: 'fitted' as const, temperature: 1 } };
    const predictions = [];
    for (const threshold of [0, 1, null]) {
      await writeManifest(dir, { ...base, thresholdRecommendation: {
        threshold, status: threshold === null ? 'target_not_met' : 'ready', targetAccuracy: 0.95,
      } });
      const model = await loadClassifier(dir);
      try {
        predictions.push(await model.predict('invoice'));
        assert.equal(model.metadata.thresholdRecommendation.threshold, threshold);
      } finally { await model.dispose(); }
    }
    assert.deepEqual(predictions[0], predictions[1]);
    assert.deepEqual(predictions[0], predictions[2]);
    const original = JSON.parse(await readFile(join(dir, 'model.json'), 'utf8'));
    for (const recommendation of [
      { threshold: 1, status: 'target_not_met', targetAccuracy: 0.95 },
      { threshold: null, status: 'ready', targetAccuracy: 0.95 },
      { threshold: NaN, status: 'ready', targetAccuracy: 0.95 },
      { threshold: null, status: 'unavailable', targetAccuracy: 0.95 },
      { threshold: null, status: 'insufficient_data', targetAccuracy: 0 },
    ]) {
      await writeFile(join(dir, 'model.json'), JSON.stringify({ ...original, thresholdRecommendation: recommendation }));
      await assert.rejects(readManifest(dir), /recommendation|threshold/);
    }
    original.thresholdRecommendation.targetAccuracy = 0.9;
    await writeFile(join(dir, 'model.json'), JSON.stringify(original));
    await assert.rejects(readManifest(dir), /checksum mismatch/);
  } finally { await rm(dir, { recursive: true, force: true }); }
});

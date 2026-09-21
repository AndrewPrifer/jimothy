import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, rename, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { train } from '../src/train.js';
import { loadClassifier } from '../src/index.js';

test('real MiniLM training and a moved self-contained bundle run with networking blocked', { timeout: 120_000 }, async () => {
  const encoder = process.env.JEV_DISTILL_ENCODER;
  assert.ok(encoder, 'Set JEV_DISTILL_ENCODER to a directory from prepare-encoder.');
  const dir = await mkdtemp(join(tmpdir(), 'jev-minilm-'));
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => { throw new Error('Network access is forbidden in the integration test'); };
  try {
    const result = await train({ task: 'examples/task.json', data: 'examples/training.jsonl', validation: 'examples/validation.jsonl',
      test: 'examples/test.jsonl', encoder, out: join(dir, 'model') });
    assert.equal(result.manifest.features.kind, 'minilm');
    assert.ok(result.report.test!.humanAccuracy! >= 0.8);
    await rename(join(dir, 'model'), join(dir, 'moved'));
    const classifier = await loadClassifier(join(dir, 'moved'));
    try {
      const states = ['My credit card was billed twice.', 'The parcel tracking says delivery is delayed.', 'The app crashes at login.'];
      const batch = await classifier.predictBatch(states);
      assert.deepEqual(batch.map(p => p.answer.type === 'choice' ? p.answer.choice : null), ['billing', 'shipping', 'technical']);
      const single = await classifier.predict(states[0]);
      assert.equal(single.answer.type, 'choice');
      if (single.answer.type === 'choice' && batch[0].answer.type === 'choice') {
        assert.ok(Math.abs(single.answer.probabilities.billing - batch[0].answer.probabilities.billing) < 0.01);
      }
      await assert.rejects(classifier.predict('hello '.repeat(300)), /wordpieces/);
      assert.deepEqual(await classifier.predictBatch([]), []);
    } finally { await classifier.dispose(); }
  } finally { globalThis.fetch = originalFetch; await rm(dir, { recursive: true, force: true }); }
});

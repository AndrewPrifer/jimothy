import test from 'node:test';
import assert from 'node:assert/strict';
import { encoderExtractor } from '../src/encoder-extractor.js';
import { ENCODER_FILES, parseManifest } from '../src/manifest.js';
import { parseTask } from '../src/schema.js';
import type { MiniLMConfig } from '../src/types.js';

test('opt-in chunking uses token windows and pools them without changing short inputs', async () => {
  const words = ['a', 'b', 'c', 'd', 'e'];
  const seen: string[] = [];
  const pipe: any = async (batch: string[]) => {
    seen.push(...batch);
    return { tolist: () => batch.map(text => text.startsWith('c') ? [0, 1] : [1, 0]) };
  };
  pipe.tokenizer = (text: string, options: { add_special_tokens?: boolean } = {}) => {
    const ids = text.trim() ? text.trim().split(/\s+/).map(word => words.indexOf(word) + 1) : [];
    return { input_ids: options.add_special_tokens === false ? ids : [-1, ...ids, -2] };
  };
  pipe.tokenizer.decode = (ids: number[]) => ids.map(id => words[id - 1]).join(' ');
  pipe.dispose = async () => {};
  const config = { kind: 'minilm', dimensions: 2, maxTokens: 4, longInput: 'chunk' } as MiniLMConfig;
  const extractor = encoderExtractor(config, pipe);
  const vectors = await extractor.encode(['a b c d e', 'c']);
  assert.deepEqual(seen, ['a b', 'c d', 'e', 'c']);
  assert.ok(Math.abs(vectors[0].values[0] - 2 / Math.sqrt(5)) < 1e-6);
  assert.ok(Math.abs(vectors[0].values[1] - 1 / Math.sqrt(5)) < 1e-6);
  assert.deepEqual(Array.from(vectors[1].values), [0, 1]);
  await assert.rejects(encoderExtractor({ ...config, longInput: undefined }, pipe).encode(['a b c']), /supports 4/);
  await extractor.dispose();

  const manifest = { format: 'jev-distill', version: 3, preprocessing: 'canonical-json-v1',
    task: parseTask({ q: { type: 'boolean', instructions: 'Relevant?' } }),
    features: { kind: 'minilm', dimensions: 384, maxTokens: 256, longInput: 'chunk', directory: 'encoder',
      modelId: 'Xenova/all-MiniLM-L6-v2', revision: 'a'.repeat(40), dtype: 'q8' },
    head: { weights: [Array(384).fill(0), Array(384).fill(0)], bias: [0, 0] },
    calibration: { method: 'temperature', status: 'insufficient_data', temperature: 1 },
    thresholdRecommendation: { status: 'insufficient_data', threshold: null, targetAccuracy: 0.95 },
    files: Object.fromEntries(['report.json', ...ENCODER_FILES.map(file => `encoder/${file}`)].map(file => [file, 'a'.repeat(64)])),
  };
  assert.equal(parseManifest(manifest).features.kind, 'minilm');
  assert.throws(() => parseManifest({ ...manifest, features: { ...manifest.features, longInput: 'truncate' } }), /Invalid MiniLM/);
});

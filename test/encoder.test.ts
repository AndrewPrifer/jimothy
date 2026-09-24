import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { join, dirname } from 'node:path';
import { tmpdir } from 'node:os';
import { prepareEncoder } from '../src/minilm.js';
import { ENCODER_FILES } from '../src/manifest.js';
import { writeManifest, readManifest } from '../src/artifact.js';
import { parseTask } from '../src/schema.js';

test('local multilingual encoder metadata sets portable dimensions, token limit and prefix', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'jimothy-encoder-'));
  const source = join(dir, 'source'), bundle = join(dir, 'bundle');
  try {
    for (const file of ENCODER_FILES) {
      await mkdir(dirname(join(source, file)), { recursive: true });
      await writeFile(join(source, file), '{}');
    }
    await writeFile(join(source, 'encoder-source.json'), JSON.stringify({ modelId: 'Xenova/multilingual-e5-small', revision: 'a'.repeat(40) }));
    await writeFile(join(source, 'config.json'), JSON.stringify({ hidden_size: 384, max_position_embeddings: 512 }));
    await writeFile(join(source, 'tokenizer_config.json'), JSON.stringify({ model_max_length: 512 }));
    await assert.rejects(prepareEncoder(join(dir, 'mismatch'), source, 'Xenova/all-MiniLM-L6-v2'), /does not match/);
    const features = await prepareEncoder(join(bundle, 'encoder'), source);
    assert.deepEqual([features.modelId, features.dimensions, features.maxTokens], ['Xenova/multilingual-e5-small', 384, 512]);
    features.inputPrefix = 'query: ';
    await writeFile(join(bundle, 'report.json'), '{}');
    const manifest = await writeManifest(bundle, {
      format: 'jev-distill', version: 3, createdAt: new Date().toISOString(),
      task: parseTask({ q: { type: 'boolean', instructions: 'Relevant?' } }), preprocessing: 'canonical-json-v1', features,
      head: { weights: [Array(384).fill(0), Array(384).fill(0)], bias: [0, 0] },
      calibration: { method: 'temperature', status: 'insufficient_data', temperature: 1 },
      thresholdRecommendation: { status: 'insufficient_data', threshold: null, targetAccuracy: 0.95 },
    });
    assert.equal((await readManifest(bundle)).features.kind, 'minilm');
    assert.equal(manifest.features.kind === 'minilm' && manifest.features.inputPrefix, 'query: ');
    const invalid = { ...manifest, head: { ...manifest.head, weights: [[], []] } };
    await writeFile(join(bundle, 'model.json'), JSON.stringify(invalid));
    await assert.rejects(readManifest(bundle), /weight dimensions/);
    const sourceMetadata = JSON.parse(await readFile(join(source, 'encoder-source.json'), 'utf8'));
    sourceMetadata.modelId = 'Xenova/all-MiniLM-L6-v2';
    await writeFile(join(source, 'encoder-source.json'), JSON.stringify(sourceMetadata));
    const legacy = await prepareEncoder(join(dir, 'legacy'), source);
    assert.equal(legacy.maxTokens, 256);
  } finally { await rm(dir, { recursive: true, force: true }); }
});

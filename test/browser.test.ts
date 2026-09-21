import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { writeManifest, readManifest } from '../src/artifact.js';
import { parseTask } from '../src/schema.js';
import { readBrowserBundle, applyWebGPUCalibration } from '../src/browser-bundle.js';
import { loadClassifier } from '../src/browser.js';
import { loadClassifier as loadNodeClassifier } from '../src/sdk.js';
import { prepareBrowser } from '../src/browser-assets.js';

test('Node and browser loaders require the current schema and verify model/assets', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'jimothy-web-'));
  const task = parseTask({ q: { type: 'boolean', instructions: 'Is it good?' } });
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async input => {
    const bytes = await readFile(join(directory, new URL(String(input)).pathname.slice(1)));
    return new Response(bytes);
  };
  try {
    await writeFile(join(directory, 'report.json'), '{}');
    const manifest = await writeManifest(directory, { format: 'jev-distill', version: 3, createdAt: new Date().toISOString(), task,
      preprocessing: 'canonical-json-v1', features: { kind: 'tfidf', vocabulary: ['good'], idf: [1], maxTokens: 4096 },
      head: { weights: [[0], [2]], bias: [0, 0] },
      thresholdRecommendation: { threshold: 0.9, status: 'ready', targetAccuracy: 0.95 },
      calibration: { method: 'temperature', status: 'fitted', temperature: 1.4 },
    });
    const browser = await readBrowserBundle('https://example.test/model.json', () => {});
    assert.deepEqual(browser.manifest, await readManifest(directory));
    assert.equal(browser.assets.size, 1);
    const invalid: [Record<string, unknown>, RegExp][] = [
      ...[1, 2, 4].map(version => [{ version }, /Unsupported model format/] as [Record<string, unknown>, RegExp]),
      [{ calibration: undefined }, /calibration must be an object/],
      [{ thresholdRecommendation: undefined }, /thresholdRecommendation must be an object/],
      [{ threshold: 0.9 }, /not acceptance policies/],
      [{ acceptance: { mode: 'automatic', status: 'ready', targetAccuracy: 0.95 } }, /not acceptance policies/],
    ];
    for (const [override, error] of invalid) {
      await writeFile(join(directory, 'model.json'), JSON.stringify({ ...manifest, ...override }));
      await assert.rejects(loadNodeClassifier(directory), error);
      await assert.rejects(readBrowserBundle('https://example.test/model.json', () => {}), error);
    }
    manifest.head.bias[0] = 1;
    await writeFile(join(directory, 'model.json'), JSON.stringify(manifest));
    await assert.rejects(readBrowserBundle('https://example.test/model.json', () => {}), /manifest checksum/);
    manifest.head.bias[0] = 0;
    await writeFile(join(directory, 'model.json'), JSON.stringify(manifest));
    await writeFile(join(directory, 'report.json'), '{"changed":true}');
    await assert.rejects(readBrowserBundle('https://example.test/model.json', () => {}), /Checksum mismatch for report/);
  } finally { globalThis.fetch = originalFetch; await rm(directory, { recursive: true, force: true }); }
});

test('FP16 calibration checks full provenance and exports advisory metadata', async () => {
  const record = JSON.parse(await readFile('benchmarks/browser-fp16-calibration.json', 'utf8'));
  const p = record.policy.provenance;
  // A minimal matching fixture avoids requiring locally downloaded encoder weights.
  const manifest = { format: 'jev-distill', version: 3, id: p.sourceModelId,
    calibration: { method: 'temperature', temperature: 1, status: 'fitted' },
    thresholdRecommendation: { threshold: 0.85, status: 'ready', targetAccuracy: 0.95 },
    features: { kind: 'minilm', modelId: p.encoder.modelId, revision: p.encoder.revision, maxTokens: 256 },
    head: { weights: [], bias: [] }, preprocessing: 'canonical-json-v1', files: { 'encoder/tokenizer.json': p.tokenizerSha256 } } as any;
  const { sha256 } = await import('../src/browser-bundle.js');
  p.headSha256 = await sha256(JSON.stringify(manifest.head));
  await assert.rejects(applyWebGPUCalibration(structuredClone(manifest), record.policy, 'wrong', p.runtime), /does not match/);
  await assert.rejects(applyWebGPUCalibration(structuredClone(manifest), record.policy, p.sourceManifestSha256, {}), /does not match/);
  await applyWebGPUCalibration(manifest, record.policy, p.sourceManifestSha256, p.runtime);
  assert.equal(manifest.calibration.temperature, record.policy.calibration.temperature);
  assert.equal(manifest.thresholdRecommendation.threshold, 0.9);
  assert.equal(manifest.id, `${p.sourceModelId}-fp16-webgpu`);
  assert.equal('acceptance' in manifest, false);
});

test('browser client isolates workers, correlates concurrent requests, cancels loading and disposes pending work', async () => {
  const descriptors = Object.fromEntries(['Worker', 'location'].map(name => [name, Object.getOwnPropertyDescriptor(globalThis, name)]));
  class FakeWorker {
    static instances: FakeWorker[] = [];
    onmessage?: (event: any) => void; onerror?: (event: any) => void; onmessageerror?: () => void;
    terminated = false; messages: any[] = [];
    constructor() { FakeWorker.instances.push(this); }
    terminate() { this.terminated = true; }
    postMessage(data: any) { this.messages.push(data); }
    reply(index: number, result: unknown) { this.onmessage?.({ data: { id: this.messages[index].id, result } }); }
  }
  Object.defineProperty(globalThis, 'Worker', { configurable: true, value: FakeWorker });
  Object.defineProperty(globalThis, 'location', { configurable: true, value: new URL('https://example.test/app/') });
  try {
    const loading = loadClassifier('/models/email');
    const worker = FakeWorker.instances.at(-1)!;
    assert.equal(worker.messages[0].args.manifestUrl, 'https://example.test/models/email/model.json');
    worker.reply(0, { id: 'local', task: { questionId: 'q' } });
    const model = await loading;
    const a = model.predict('a'), b = model.predict('b');
    worker.reply(2, [{ answer: 'b' }]); worker.reply(1, [{ answer: 'a' }]);
    assert.deepEqual(await Promise.all([a, b]), [{ answer: 'a' }, { answer: 'b' }]);
    const pending = model.predict('c');
    await model.dispose(); await model.dispose();
    await assert.rejects(pending, /disposed/);
    await assert.rejects(model.predict('d'), /disposed/);
    assert.ok(worker.terminated);
    const abort = new AbortController();
    const cancelled = loadClassifier('/model', { signal: abort.signal }); abort.abort();
    await assert.rejects(cancelled, { name: 'AbortError' });
    assert.ok(FakeWorker.instances.at(-1)!.terminated);
    const failed = loadClassifier('/model'); FakeWorker.instances.at(-1)!.onerror?.({ message: 'Worker unavailable' });
    await assert.rejects(failed, /Worker unavailable/);
    assert.ok(FakeWorker.instances.at(-1)!.terminated);
    await assert.rejects(loadClassifier('/model', { assetsUrl: 'https://other.test/assets' }), /same origin/);
    await assert.rejects(loadClassifier('/model', { device: 'webgpu' }), /calibrated FP16/);
  } finally {
    for (const [key, descriptor] of Object.entries(descriptors)) {
      if (descriptor) Object.defineProperty(globalThis, key, descriptor);
      else Reflect.deleteProperty(globalThis, key);
    }
  }
});

test('prepare-browser refuses to replace an existing directory', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'jimothy-assets-'));
  try {
    await writeFile(join(directory, 'keep.txt'), 'keep');
    await assert.rejects(prepareBrowser(directory), /already exists/);
    assert.equal(await readFile(join(directory, 'keep.txt'), 'utf8'), 'keep');
  } finally { await rm(directory, { recursive: true, force: true }); }
});

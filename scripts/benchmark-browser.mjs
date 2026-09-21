#!/usr/bin/env node
// A local experiment, not the browser SDK. Run from the repository root.
import { createServer } from 'node:http';
import { createReadStream } from 'node:fs';
import { mkdir, readFile, writeFile, stat, rename } from 'node:fs/promises';
import { resolve, join, extname } from 'node:path';
import { createHash, randomUUID } from 'node:crypto';
import { cpus, platform, arch, release } from 'node:os';
import { loadClassifier } from '../dist/index.js';

const model = resolve('models/banking77-auto-v2/minilm');
const cache = resolve('.cache/browser-benchmark');
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
const json = async path => JSON.parse(await readFile(path, 'utf8'));
const source = await json(join(model, 'encoder/encoder-source.json'));
const modelBytes = await readFile(join(model, 'model.json'));
const testBytes = await readFile('datasets/banking77/test.jsonl');
const fingerprint = { modelSha256: sha256(modelBytes), testSha256: sha256(testBytes) };

if (process.argv[2] === 'prepare') {
  await mkdir(join(cache, 'onnx'), { recursive: true });
  if (process.argv.includes('--download')) {
    for (const file of ['model_fp16.onnx', 'model.onnx']) {
      const path = join(cache, 'onnx', file);
      try { await stat(path); continue; } catch {}
      const url = `https://huggingface.co/${source.modelId}/resolve/${source.revision}/onnx/${file}`;
      console.log(`Downloading ${url}`);
      const response = await fetch(url, { signal: AbortSignal.timeout(180_000) });
      if (!response.ok) throw new Error(`Download failed: HTTP ${response.status}`);
      await writeFile(`${path}.partial`, new Uint8Array(await response.arrayBuffer()));
      await rename(`${path}.partial`, path);
    }
  }
  console.log('Generating native q8 reference predictions for all test inputs...');
  const rows = testBytes.toString().trim().split('\n').map(JSON.parse);
  const classifier = await loadClassifier(model);
  try {
    const cutoff = classifier.metadata.thresholdRecommendation.threshold;
    const predictions = await classifier.predictBatch(rows.map(row => row.state));
    await writeFile(join(cache, 'reference.json'), JSON.stringify({
      ...fingerprint, modelId: classifier.metadata.id, source,
      environment: { node: process.version, cpu: cpus()[0]?.model, platform: platform(), arch: arch(), os: release() },
      rows: rows.map((row, i) => ({ id: row.id, text: row.state, label: row.label,
        reference: { choice: predictions[i].answer.choice, accepted: cutoff !== null && predictions[i].maxProbability >= cutoff,
          probabilities: predictions[i].answer.probabilities } })),
    }));
  } finally { await classifier.dispose(); }
  console.log(`Prepared ${rows.length} reference predictions in ${cache}`);
} else if (process.argv[2] === 'serve') {
  const reference = await json(join(cache, 'reference.json'));
  if (Object.keys(fingerprint).some(key => reference[key] !== fingerprint[key])) {
    throw new Error('Model or dataset changed; run prepare again.');
  }
  const token = randomUUID();
  const port = Number(process.env.BENCHMARK_PORT ?? 4317);
  const origin = `http://127.0.0.1:${port}`;
  const directory = resolve('models/browser-benchmark', new Date().toISOString().replaceAll(':', '-'));
  await mkdir(directory, { recursive: true });
  const files = new Map([
    ['/', resolve('scripts/browser-benchmark/index.html')],
    ['/app.js', resolve('scripts/browser-benchmark/app.js')],
    ['/worker.js', resolve('scripts/browser-benchmark/worker.js')],
    ['/linear.js', resolve('dist/linear.js')],
    ['/model.json', join(model, 'model.json')],
    ['/reference.json', join(cache, 'reference.json')],
    ['/vendor/transformers.min.js', resolve('node_modules/@huggingface/transformers/dist/transformers.min.js')],
  ]);
  for (const file of ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json']) {
    files.set(`/encoder/${file}`, join(model, 'encoder', file));
  }
  files.set('/encoder/onnx/model_quantized.onnx', join(model, 'encoder/onnx/model_quantized.onnx'));
  for (const file of ['model_fp16.onnx', 'model.onnx']) files.set(`/encoder/onnx/${file}`, join(cache, 'onnx', file));
  for (const flavor of ['', '.jsep', '.asyncify', '.jspi']) {
    for (const extension of ['mjs', 'wasm']) {
      const file = `ort-wasm-simd-threaded${flavor}.${extension}`;
      files.set(`/vendor/${file}`, resolve('node_modules/onnxruntime-web/dist', file));
    }
  }
  const assets = {};
  for (const [url, path] of files) {
    if (!url.endsWith('.onnx') && !url.startsWith('/vendor/')) continue;
    try { const bytes = await readFile(path); assets[url] = { bytes: bytes.length, sha256: sha256(bytes) }; } catch {}
  }
  const config = { token, assets, source, fingerprint,
    versions: Object.fromEntries(await Promise.all(['@huggingface/transformers', 'onnxruntime-web', 'onnxruntime-node'].map(async name => [name, (await json(`node_modules/${name}/package.json`)).version]))),
    environment: reference.environment };
  const mime = { '.js': 'text/javascript', '.mjs': 'text/javascript', '.html': 'text/html', '.json': 'application/json', '.wasm': 'application/wasm' };
  const server = createServer(async (req, res) => {
    res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
    res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
    res.setHeader('Cache-Control', 'no-store');
    res.setHeader('X-Content-Type-Options', 'nosniff');
    // Only the loopback app can submit results; never expose arbitrary workspace paths.
    if (req.headers.host !== `127.0.0.1:${port}`) { res.writeHead(403).end(); return; }
    const url = new URL(req.url, origin);
    try {
      if (req.method === 'POST' && url.pathname === '/results' && req.headers.origin === origin && req.headers['x-benchmark-token'] === token) {
        let body = '';
        for await (const chunk of req) { body += chunk; if (body.length > 2_000_000) throw new Error('Result too large'); }
        const result = JSON.parse(body);
        const destination = join(directory, `${Date.now()}-${randomUUID().slice(0, 8)}.json`);
        await writeFile(destination, JSON.stringify({ ...result, server: { ...config, token: undefined } }, null, 2) + '\n');
        console.log(`Saved ${destination}`);
        res.setHeader('Content-Type', 'application/json'); res.end(JSON.stringify({ saved: destination })); return;
      }
      if (req.method !== 'GET') { res.writeHead(405).end(); return; }
      if (url.pathname === '/config.json') { res.setHeader('Content-Type', 'application/json'); res.end(JSON.stringify(config)); return; }
      const path = files.get(url.pathname);
      if (!path) { res.writeHead(404).end(); return; }
      const info = await stat(path);
      res.setHeader('Content-Type', mime[extname(path)] ?? 'application/octet-stream');
      res.setHeader('Content-Length', info.size);
      createReadStream(path).pipe(res);
    } catch (error) { res.writeHead(400).end(String(error.message)); }
  });
  server.listen(port, '127.0.0.1', () => console.log(`Benchmark: ${origin}\nResults: ${directory}`));
} else {
  console.log('Usage: node scripts/benchmark-browser.mjs prepare [--download] | serve');
  process.exitCode = 1;
}

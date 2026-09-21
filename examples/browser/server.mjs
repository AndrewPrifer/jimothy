#!/usr/bin/env node
// Serves only static inference assets. User inputs never reach this server.
import { createServer } from 'node:http';
import { createReadStream } from 'node:fs';
import { readFile, stat } from 'node:fs/promises';
import { dirname, resolve, join, extname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { browserAssets } from '../../dist/browser-assets.js';
import { readManifest } from '../../dist/artifact.js';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '../..');
const sha256 = value => createHash('sha256').update(value).digest('hex');
const readJSON = async path => JSON.parse(await readFile(path, 'utf8'));
const modelPath = resolve(root, process.env.DEMO_MODEL_DIR ?? 'models/banking77-auto-v2/minilm');
let manifest;
try { manifest = await readManifest(modelPath); }
catch (error) { throw new Error(`Cannot load the demo model at ${modelPath}. Prepare a MiniLM bundle first; see examples/browser/README.md. ${error.message}`); }
if (manifest.features.kind !== 'minilm' || manifest.task.question.type !== 'choice') throw new Error('This example requires a MiniLM choice classifier.');
const browser = await browserAssets();
const versions = browser.versions;
const originalPolicy = { calibration: manifest.calibration, thresholdRecommendation: manifest.thresholdRecommendation };
const fp16Path = join(root, '.cache/browser-benchmark/onnx/model_fp16.onnx');
let fp16 = null, gpuUnavailableReason, gpuPolicy;
try {
  const record = await readJSON(join(root, 'benchmarks/browser-fp16-calibration.json'));
  const policy = record.policy, p = policy.provenance;
  if (!record.complete || policy.format !== 'jev-distill-browser-policy-experiment' || policy.version !== 1 ||
      sha256(JSON.stringify(policy, null, 2) + '\n') !== record.policySha256 ||
      policy.acceptance.status !== 'ready' || policy.calibration.status !== 'fitted' || p.device !== 'webgpu' || p.encoder.dtype !== 'fp16' ||
      p.sourceManifestSha256 !== sha256(await readFile(join(modelPath, 'model.json'))) ||
      p.headSha256 !== sha256(JSON.stringify(manifest.head)) ||
      p.encoder.sha256 !== sha256(await readFile(fp16Path)) ||
      p.tokenizerSha256 !== sha256(await readFile(join(modelPath, 'encoder/tokenizer.json'))) ||
      Object.entries(versions).some(([key, version]) => p.runtime[key] !== version)) throw new Error('The FP16 assets or runtime do not match the validated calibration.');
  gpuPolicy = policy;
  fp16 = { calibration: policy.calibration, thresholdRecommendation: { threshold: policy.threshold, status: 'ready', targetAccuracy: policy.acceptance.targetAccuracy },
    encoderSha256: p.encoder.sha256 };
} catch (error) {
  gpuUnavailableReason = 'A matching calibrated FP16 export is not available for this model. WebAssembly is ready to use.';
  console.warn(`WebGPU option disabled: ${error.message}`);
}
const fingerprint = sha256(JSON.stringify({ manifest, fp16, versions, worker: sha256(await readFile(join(root, 'dist/browser-worker.js'))) })).slice(0, 24);
const base = `/assets/${fingerprint}/`;
const assetsUrl = `${base}sdk/`;
const files = new Map([
  ['/', join(here, 'index.html')], ['/style.css', join(here, 'style.css')], ['/app.js', join(here, 'app.js')],
  ['/file-help.js', join(here, 'file-help.js')], ['/inference.js', join(here, 'inference.js')],
  ['/jimothy/browser.js', join(root, 'dist/browser.js')],
  [`${base}model.json`, join(modelPath, 'model.json')],
]);
for (const file of Object.keys(manifest.files)) files.set(`${base}${file}`, join(modelPath, file));
for (const [name, source] of browser.files) files.set(`${assetsUrl}${name}`, source);
if (fp16) files.set(`${base}encoder/onnx/model_fp16.onnx`, fp16Path);
const webgpu = fp16 ? { modelUrl: `${base}encoder/onnx/model_fp16.onnx`, policyUrl: `${base}fp16-policy.json` } : undefined;
const config = { base, assetsUrl, webgpu, modelId: manifest.id, labels: manifest.task.labels, maxTokens: manifest.features.maxTokens,
  taskName: manifest.task.labels.length === 77 && manifest.task.labels.includes('card_arrival') ? 'BANKING77 · Banking support' : 'Custom classifier',
  policies: { wasm: originalPolicy, webgpu: fp16 }, gpuUnavailableReason, versions };
const port = Number(process.env.DEMO_PORT ?? 4319);
const mime = { '.html': 'text/html; charset=utf-8', '.css': 'text/css', '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json', '.wasm': 'application/wasm' };
const server = createServer(async (req, res) => {
  res.setHeader('Cross-Origin-Opener-Policy', 'same-origin'); res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.setHeader('Content-Security-Policy', "default-src 'self'; script-src 'self' 'wasm-unsafe-eval' blob:; style-src 'self'; worker-src 'self' blob:; connect-src 'self' blob:; img-src 'self' data:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'");
  if (![ `127.0.0.1:${port}`, `localhost:${port}` ].includes(req.headers.host)) { res.writeHead(403).end(); return; }
  if (!['GET', 'HEAD'].includes(req.method)) { res.writeHead(405).end(); return; }
  try {
    const pathname = new URL(req.url, `http://127.0.0.1:${port}`).pathname;
    const json = pathname === '/config.json' ? config : pathname === `${assetsUrl}versions.json` ? versions : pathname === `${base}fp16-policy.json` ? gpuPolicy : undefined;
    if (json) {
      res.setHeader('Cache-Control', 'no-store'); res.setHeader('Content-Type', 'application/json');
      res.end(req.method === 'HEAD' ? undefined : JSON.stringify(json)); return;
    }
    const file = files.get(pathname);
    if (!file) { res.writeHead(404).end('Not found'); return; }
    const info = await stat(file);
    res.setHeader('Cache-Control', pathname.startsWith(base) ? 'public, max-age=31536000, immutable' : 'no-store');
    res.setHeader('Content-Type', mime[extname(file)] ?? 'application/octet-stream'); res.setHeader('Content-Length', info.size);
    if (req.method === 'HEAD') { res.end(); return; }
    const stream = createReadStream(file);
    stream.on('error', () => res.destroy()); stream.pipe(res);
  } catch { res.writeHead(404).end('Asset unavailable'); }
});
server.on('error', error => { console.error(`Cannot start the demo: ${error.message}. Set DEMO_PORT to use another port.`); process.exitCode = 1; });
server.listen(port, '127.0.0.1', () => console.log(`\nBrowser playground: http://127.0.0.1:${port}\nModel: ${manifest.id}\nWebGPU FP16: ${fp16 ? 'calibrated and available' : 'unavailable; using q8 WASM'}\nPress Ctrl-C to stop.\n`));

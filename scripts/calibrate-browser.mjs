#!/usr/bin/env node
// Fixed BANKING77 deployment experiment. No encoder/head training or teacher calls.
import { createServer } from 'node:http';
import { createReadStream } from 'node:fs';
import { readFile, writeFile, mkdir, stat } from 'node:fs/promises';
import { resolve, join, extname, dirname } from 'node:path';
import { createHash, randomUUID } from 'node:crypto';
import { readManifest } from '../dist/artifact.js';
import { loadDataset } from '../dist/train.js';
import { splitDevelopment, datasetDigest, assertDisjoint } from '../dist/data.js';
import { fitBrowserPolicy, validateLogits, evaluatePolicy, verifyBrowserDecisions } from './browser-calibration/analysis.mjs';

const json = async path => JSON.parse(await readFile(path, 'utf8'));
const hash = value => createHash('sha256').update(value).digest('hex');
const fileHash = async path => hash(await readFile(path));
const modelPath = resolve('models/banking77-auto-v2/minilm');
const manifest = await readManifest(modelPath);
const originalReport = await json(join(modelPath, 'report.json'));
if (manifest.features.kind !== 'minilm') throw new Error('This experiment requires a MiniLM model.');
const [train, validation, test] = await Promise.all(['train', 'validation', 'test'].map(async split =>
  (await loadDataset({ data: `datasets/banking77/${split}.jsonl` }, manifest.task)).examples));
assertDisjoint(train, validation, 'Development'); assertDisjoint(train, test, 'Test'); assertDisjoint(validation, test, 'Test');
const development = splitDevelopment(validation, originalReport.seed);
for (const [name, examples] of Object.entries({ train, validation, test })) {
  if (datasetDigest(examples) !== originalReport.dataset[name].sha256) throw new Error(`Changed ${name} dataset.`);
}
for (const [name, examples] of Object.entries(development)) {
  if (datasetDigest(examples) !== originalReport.dataset.development[name].sha256) throw new Error(`Changed ${name} development split.`);
}
const encoderPath = resolve('.cache/browser-benchmark/onnx/model_fp16.onnx');
const encoderSha256 = await fileHash(encoderPath);
const benchmark = await json('benchmarks/browser-minilm-m3-max.json');
if (encoderSha256 !== benchmark.server.assets['/encoder/onnx/model_fp16.onnx'].sha256) throw new Error('FP16 encoder differs from the benchmarked export.');
const versions = Object.fromEntries(await Promise.all(['@huggingface/transformers', 'onnxruntime-web'].map(async name => [name, (await json(`node_modules/${name}/package.json`)).version])));
const output = resolve(process.argv[2] ?? `models/browser-fp16-calibration/${new Date().toISOString().replaceAll(':', '-')}`);
await mkdir(dirname(output), { recursive: true });
await mkdir(output, { recursive: false });
const token = randomUUID();
const port = Number(process.env.CALIBRATION_PORT ?? 4318);
const origin = `http://127.0.0.1:${port}`;
const provenance = { sourceModelId: manifest.id, sourceManifestSha256: await fileHash(join(modelPath, 'model.json')),
  headSha256: hash(JSON.stringify(manifest.head)), encoder: { modelId: manifest.features.modelId, revision: manifest.features.revision,
    dtype: 'fp16', sha256: encoderSha256 }, tokenizerSha256: await fileHash(join(modelPath, 'encoder/tokenizer.json')),
  runtime: versions, device: 'webgpu', preprocessing: manifest.preprocessing, pooling: 'mean', normalize: true,
  maxTokens: manifest.features.maxTokens, calibrationBatchSize: 32,
  datasets: { ...originalReport.dataset, note: 'Previously inspected BANKING77 test set; parameters are frozen before this evaluation.' } };
const publicRows = examples => examples.map(e => ({ id: e.id, text: e.text }));
const files = new Map([
  ['/', resolve('scripts/browser-calibration/index.html')], ['/app.js', resolve('scripts/browser-calibration/app.js')],
  ['/worker.js', resolve('scripts/browser-calibration/worker.js')], ['/linear.js', resolve('dist/linear.js')],
  ['/model.json', join(modelPath, 'model.json')], ['/encoder/onnx/model_fp16.onnx', encoderPath],
  ['/vendor/transformers.min.js', resolve('node_modules/@huggingface/transformers/dist/transformers.min.js')],
]);
for (const file of ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json']) files.set(`/encoder/${file}`, join(modelPath, 'encoder', file));
for (const flavor of ['', '.jsep', '.asyncify', '.jspi']) for (const extension of ['mjs', 'wasm']) {
  const file = `ort-wasm-simd-threaded${flavor}.${extension}`;
  files.set(`/vendor/${file}`, resolve('node_modules/onnxruntime-web/dist', file));
}
const mime = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json', '.wasm': 'application/wasm' };
let frozen, policySha256, fitting = false;
const evaluations = {}, testDecisions = {};
const report = { format: 'jev-distill-browser-calibration-experiment', version: 1, createdAt: new Date().toISOString(), provenance,
  originalPolicy: { calibration: manifest.calibration, threshold: manifest.thresholdRecommendation.threshold ?? 1,
    acceptance: { mode: 'automatic', status: manifest.thresholdRecommendation.status, targetAccuracy: manifest.thresholdRecommendation.targetAccuracy } },
  protocol: 'Fit temperature on the original calibration split; select threshold on the original acceptance split with the existing corrected grid. Freeze policy before reading test inputs. Evaluate unchanged policy on batch-32 and single-input WebGPU inference.' };
const send = (res, body) => { res.setHeader('Content-Type', 'application/json'); res.end(JSON.stringify(body)); };
const server = createServer(async (req, res) => {
  res.setHeader('Cross-Origin-Opener-Policy', 'same-origin'); res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
  res.setHeader('Cache-Control', 'no-store'); res.setHeader('X-Content-Type-Options', 'nosniff');
  if (req.headers.host !== `127.0.0.1:${port}`) { res.writeHead(403).end(); return; }
  try {
    const path = new URL(req.url, origin).pathname;
    if (req.method === 'POST') {
      if (req.headers.origin !== origin || req.headers['x-calibration-token'] !== token) { res.writeHead(403).end(); return; }
      let raw = '';
      for await (const chunk of req) { raw += chunk; if (raw.length > 12_000_000) throw new Error('Request too large.'); }
      const body = JSON.parse(raw);
      if (path === '/fit') {
        if (frozen || fitting) throw new Error('The policy is already frozen; it cannot be refitted after test access.');
        fitting = true;
        try {
          const fitted = fitBrowserPolicy(body.calibration, body.acceptance, development, manifest);
          const policy = { format: 'jev-distill-browser-policy-experiment', version: 1, createdAt: new Date().toISOString(),
            provenance, browser: body.browser, calibration: fitted.calibration, threshold: fitted.threshold, acceptance: fitted.acceptance };
          const serialized = JSON.stringify(policy, null, 2) + '\n';
          await writeFile(join(output, 'policy.json'), serialized, { flag: 'wx' });
          policySha256 = hash(serialized);
          frozen = await json(join(output, 'policy.json'));
          report.policySha256 = policySha256; report.policy = frozen; report.calibrationFit = fitted.fit; report.acceptanceSelection = fitted.selection;
          await writeFile(join(output, 'development-logits.json'), JSON.stringify(body) + '\n', { flag: 'wx' });
          await writeFile(join(output, 'report.json'), JSON.stringify(report, null, 2) + '\n');
          console.log(`Frozen FP16 policy: T=${frozen.calibration.temperature}, threshold=${frozen.threshold}, status=${frozen.acceptance.status}`);
          send(res, { policySha256, calibration: frozen.calibration, threshold: frozen.threshold, acceptance: frozen.acceptance }); return;
        } finally { fitting = false; }
      }
      if (path === '/evaluate') {
        if (!frozen || body.policySha256 !== policySha256) throw new Error('Freeze and load the saved policy before evaluation.');
        if (!['batch32', 'single'].includes(body.mode) || evaluations[body.mode]) throw new Error('Invalid or already evaluated mode.');
        const logits = validateLogits(body.rows, test, manifest.task.labels.length);
        const current = evaluatePolicy(logits, test, manifest.task.labels, frozen);
        verifyBrowserDecisions(body.rows, current.decisions);
        const before = evaluatePolicy(logits, test, manifest.task.labels, report.originalPolicy);
        const uncalibrated = evaluatePolicy(logits, test, manifest.task.labels, { ...report.originalPolicy, calibration: { temperature: 1 } });
        evaluations[body.mode] = { oldQ8CalibrationOnFP16: before.summary, recalibratedFP16: current.summary,
          uncalibrated: { logLoss: uncalibrated.summary.logLoss, brier: uncalibrated.summary.brier, ece10: uncalibrated.summary.ece10 },
          changedChoices: current.decisions.filter((p, i) => p.choice !== before.decisions[i].choice).length,
          changedAcceptance: current.decisions.filter((p, i) => p.accepted !== before.decisions[i].accepted).length,
          browserPolicyApplicationVerified: true, elapsedMs: body.elapsedMs };
        testDecisions[body.mode] = current.decisions;
        if (testDecisions.single && testDecisions.batch32) report.batchingStability = {
          examples: test.length, changedChoices: testDecisions.single.filter((p, i) => p.choice !== testDecisions.batch32[i].choice).length,
          changedAcceptance: testDecisions.single.filter((p, i) => p.accepted !== testDecisions.batch32[i].accepted).length,
          maxConfidenceDifference: Math.max(...testDecisions.single.map((p, i) => Math.abs(p.maxProbability - testDecisions.batch32[i].maxProbability))),
        };
        report.evaluation = evaluations;
        report.complete = !!(evaluations.single && evaluations.batch32);
        await writeFile(join(output, `${body.mode}-test-logits.json`), JSON.stringify(body) + '\n', { flag: 'wx' });
        await writeFile(join(output, 'report.json'), JSON.stringify(report, null, 2) + '\n');
        console.log(`${body.mode}: ${JSON.stringify(current.summary)}`);
        send(res, { ...evaluations[body.mode], batchingStability: report.batchingStability, saved: output }); return;
      }
      res.writeHead(404).end(); return;
    }
    if (req.method !== 'GET') { res.writeHead(405).end(); return; }
    if (path === '/config.json') { send(res, { token, versions }); return; }
    if (path === '/development.json') { send(res, { calibration: publicRows(development.calibration), acceptance: publicRows(development.acceptance) }); return; }
    if (path === '/test.json' || path === '/policy.json') {
      if (!frozen) throw new Error('Policy must be frozen before test access.');
      send(res, path === '/test.json' ? publicRows(test) : { policy: await json(join(output, 'policy.json')), policySha256 }); return;
    }
    const file = files.get(path);
    if (!file) { res.writeHead(404).end(); return; }
    const info = await stat(file);
    res.setHeader('Content-Type', mime[extname(file)] ?? 'application/octet-stream'); res.setHeader('Content-Length', info.size);
    createReadStream(file).pipe(res);
  } catch (error) { res.writeHead(400); send(res, { error: error.message }); }
});
server.listen(port, '127.0.0.1', () => console.log(`Calibrate FP16 WebGPU: ${origin}\nArtifacts: ${output}`));

// Run against a separately installed npm tarball, not the source checkout.
import { createServer } from 'node:http';
import { createReadStream } from 'node:fs';
import { cp, mkdir, mkdtemp, stat, writeFile } from 'node:fs/promises';
import { extname, join, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';
import { execFile } from 'node:child_process';
import { build } from 'esbuild';
import { writeManifest } from '../../dist/artifact.js';
import { parseTask } from '../../dist/schema.js';
import { loadClassifier } from '../../dist/index.js';

if (!process.argv[2]) throw new Error('Usage: node scripts/browser-sdk/server.mjs /absolute/path/to/consumer-with-jimothy-installed');
const consumer = resolve(process.argv[2]);
const packageDist = join(consumer, 'node_modules/jimothy/dist');
const work = await mkdtemp(join(consumer, 'smoke-'));
const publicDir = join(work, 'public');
await mkdir(publicDir);
await promisify(execFile)(process.execPath, [join(packageDist, 'cli.js'), 'prepare-browser', '--out', join(publicDir, 'jimothy')], { cwd: consumer });
const source = join(work, 'app.js');
await cp(fileURLToPath(new URL('app.js', import.meta.url)), source);
const built = await build({ entryPoints: [source], bundle: true, platform: 'browser', format: 'esm', outfile: join(publicDir, 'app.js'), metafile: true });
if (Object.keys(built.metafile.inputs).some(file => /transformers|onnxruntime|dist\/(cli|sdk)\.js/.test(file))) throw new Error('Browser entry leaked Node or encoder dependencies.');
const states = ['good', 'bad', { subject: 'good', body: ['bad', 'good'] }], fixtures = [];
for (const type of ['choice', 'boolean', 'noul', 'score']) {
  const task = parseTask({ q: { type, instructions: 'Classify the message.', ...(type === 'choice' ? { criteria: { bad: 'Bad', good: 'Good' } } : type === 'score' ? { criteria: ['Bad', 'Good'] } : {}) } });
  const directory = join(publicDir, 'models', type);
  await mkdir(directory, { recursive: true }); await writeFile(join(directory, 'report.json'), '{}');
  await writeManifest(directory, { format: 'jev-distill', version: 3, createdAt: new Date().toISOString(), task,
    preprocessing: 'canonical-json-v1', features: { kind: 'tfidf', vocabulary: ['bad', 'good'], idf: [1, 1], maxTokens: 4096 },
    head: { weights: [[2, 0], [0, 2]], bias: [0, 0] }, calibration: { method: 'temperature', status: 'fitted', temperature: 1.3 },
    thresholdRecommendation: { threshold: 0.8, status: 'ready', targetAccuracy: 0.95 } });
  const classifier = await loadClassifier(directory);
  fixtures.push({ name: type, states, predictions: await classifier.predictBatch(states), metadata: classifier.metadata });
  await classifier.dispose();
}
await cp(join(publicDir, 'models/choice'), join(publicDir, 'models/damaged'), { recursive: true });
await writeFile(join(publicDir, 'models/damaged/report.json'), '{"damaged":true}');
const emailPath = resolve('models/email-300/minilm');
try {
  await stat(emailPath);
  await cp(emailPath, join(publicDir, 'models/email'), { recursive: true });
  const classifier = await loadClassifier(emailPath);
  const samples = [
    { from: 'shop@example.com', subject: 'Your order', body: 'We received your payment of $24. Your order will ship tomorrow.' },
    { from: 'alex@example.com', subject: 'Lunch tomorrow?', body: 'Are you free to catch up for lunch at noon?' },
    { from: 'store@example.com', subject: 'Summer sale', body: 'Get 20% off all orders this weekend. Shop now.' },
  ];
  fixtures.push({ name: 'email', states: samples, predictions: await classifier.predictBatch(samples), metadata: classifier.metadata });
  await classifier.dispose();
} catch (error) { if (error.code !== 'ENOENT') throw error; console.log('Email bundle absent; testing TF-IDF only.'); }
await writeFile(join(publicDir, 'fixtures.json'), JSON.stringify(fixtures));
await writeFile(join(publicDir, 'index.html'), '<!doctype html><meta charset="utf-8"><title>Testing browser SDK</title><pre>Running packed-package browser tests…\n</pre><script type="module" src="/app.js"></script>');
const mime = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json', '.wasm': 'application/wasm' };
const port = Number(process.env.SDK_TEST_PORT ?? 4333);
const server = createServer(async (req, res) => {
  if (!['GET', 'HEAD'].includes(req.method)) { res.writeHead(405).end(); return; }
  const pathname = new URL(req.url, `http://127.0.0.1:${port}`).pathname;
  const file = resolve(publicDir, `.${pathname === '/' ? '/index.html' : pathname}`);
  if (!file.startsWith(publicDir + sep)) { res.writeHead(403).end(); return; }
  try {
    const info = await stat(file);
    if (!info.isFile()) { res.writeHead(404).end(); return; }
    // Deliberately no COOP/COEP: the one-thread SDK must work on ordinary static hosts.
    res.writeHead(200, { 'Content-Type': mime[extname(file)] ?? 'application/octet-stream', 'Content-Length': info.size, 'Cache-Control': 'no-store' });
    if (req.method === 'HEAD') res.end(); else createReadStream(file).pipe(res);
  } catch { res.writeHead(404).end(); }
});
server.listen(port, '127.0.0.1', () => console.log(`Packed-package browser tests: http://127.0.0.1:${port}\nConsumer: ${consumer}\nAssets: ${publicDir}`));

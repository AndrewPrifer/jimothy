#!/usr/bin/env node
import { createServer } from 'node:http';
import { createReadStream } from 'node:fs';
import { readFile, stat } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { dirname, resolve, join, extname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { browserAssets } from '../../dist/browser-assets.js';
import { readManifest } from '../../dist/artifact.js';
import { object, targetFromAnswer } from '../../dist/schema.js';
import { teacherEndpoint } from '../../dist/teacher.js';
import { emailState } from './email.js';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '../..');
const mime = { '.html': 'text/html; charset=utf-8', '.css': 'text/css', '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json', '.wasm': 'application/wasm' };
const MAX_BODY_BYTES = 400_000;
const send = (res, status, body) => {
  res.writeHead(status, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' });
  res.end(JSON.stringify(body));
};

/** Only this handler holds the key. No email or provider response is written to disk or logs. */
export function createHandler({ manifest, config, files, apiKey, teacher, endpoint, port, fetchImpl = fetch }) {
  let pending = false, retryAt = 0;
  return async (req, res) => {
    res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
    res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
    res.setHeader('X-Content-Type-Options', 'nosniff');
    res.setHeader('Content-Security-Policy', "default-src 'self'; script-src 'self' 'wasm-unsafe-eval' blob:; style-src 'self'; worker-src 'self' blob:; connect-src 'self' blob:; img-src 'self' data:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'");
    const host = req.headers.host;
    if (![ `127.0.0.1:${port()}`, `localhost:${port()}` ].includes(host)) { send(res, 403, { error: 'Invalid host.' }); return; }
    const pathname = new URL(req.url, `http://${host}`).pathname;
    if (pathname === '/api/jev') {
      if (req.method !== 'POST') { send(res, 405, { error: 'Use POST.' }); return; }
      // Block other websites from using this local, authenticated proxy.
      if (req.headers.origin !== `http://${host}` ||
          (req.headers['sec-fetch-site'] && req.headers['sec-fetch-site'] !== 'same-origin')) {
        send(res, 403, { error: 'Use the email playground on this server.' }); return;
      }
      if (req.headers['content-type']?.split(';')[0].trim() !== 'application/json') { send(res, 415, { error: 'Send JSON.' }); return; }
      if (!apiKey) { send(res, 503, { error: 'Set AI_GATEWAY_API_KEY on the server to use Jev.' }); return; }
      if (pending) { send(res, 429, { error: 'A Jev request is already running. Try again shortly.' }); return; }
      if (Date.now() < retryAt) {
        const seconds = Math.ceil((retryAt - Date.now()) / 1000);
        res.setHeader('Retry-After', String(seconds));
        send(res, 429, { error: `Jev is rate limited. Try again in ${seconds}s.` }); return;
      }
      let state;
      try {
        if (Number(req.headers['content-length']) > MAX_BODY_BYTES) { send(res, 413, { error: 'Email is too long.' }); return; }
        const chunks = []; let bytes = 0;
        for await (const chunk of req) {
          bytes += chunk.length;
          if (bytes > MAX_BODY_BYTES) { send(res, 413, { error: 'Email is too long.' }); return; }
          chunks.push(chunk);
        }
        state = emailState(JSON.parse(Buffer.concat(chunks).toString('utf8')));
      } catch { send(res, 400, { error: 'Provide from, subject, and a non-empty body as text.' }); return; }
      // Recheck after the asynchronous body read so parallel submissions cannot race the lock.
      if (pending) { send(res, 429, { error: 'A Jev request is already running. Try again shortly.' }); return; }
      pending = true;
      const abort = new AbortController();
      res.on('close', () => { if (!res.writableEnded) abort.abort(); });
      try {
        const response = await fetchImpl(endpoint, {
          method: 'POST', redirect: 'error', signal: AbortSignal.any([abort.signal, AbortSignal.timeout(30_000)]),
          headers: { Authorization: `Bearer ${apiKey}`, 'Content-Type': 'application/json' },
          body: JSON.stringify({ model: teacher, state, questions: { [manifest.task.questionId]: manifest.task.question } }),
        });
        if (!response.ok) {
          await response.body?.cancel();
          if (response.status === 429) {
            const after = response.headers.get('retry-after');
            const seconds = after && Number.isFinite(Number(after)) ? Number(after) : (Date.parse(after ?? '') - Date.now()) / 1000;
            const wait = Math.max(1, Math.ceil(Number.isFinite(seconds) ? seconds : 30));
            retryAt = Date.now() + wait * 1000;
            res.setHeader('Retry-After', String(wait));
            send(res, 429, { error: `Jev is rate limited. Try again in ${wait}s.` });
          } else send(res, 502, { error: [401, 403].includes(response.status)
            ? 'Jev rejected the server API key. Check AI_GATEWAY_API_KEY.' : `Jev returned HTTP ${response.status}. Try again.` });
          return;
        }
        let probabilities, choice;
        try {
          const raw = object(await response.json(), 'Teacher response');
          const answer = object(object(raw.answers, 'answers')[manifest.task.questionId], 'Answer');
          const values = targetFromAnswer(answer, manifest.task);
          probabilities = Object.fromEntries(manifest.task.labels.map((label, i) => [label, values[i]]));
          choice = answer.choice ?? manifest.task.labels[values.indexOf(Math.max(...values))];
          if (!manifest.task.labels.includes(choice)) throw new Error('Invalid choice.');
        } catch { send(res, 502, { error: 'Jev returned an invalid classification. Try again.' }); return; }
        // Return only the classification, never credentials, headers, or provider routing metadata.
        send(res, 200, { model: teacher, answers: { [manifest.task.questionId]: { type: 'choice', choice, probabilities } } });
      } catch {
        if (!res.destroyed) send(res, 504, { error: 'Jev did not respond. Try again.' });
      } finally { pending = false; }
      return;
    }
    if (!['GET', 'HEAD'].includes(req.method)) { send(res, 405, { error: 'Method not allowed.' }); return; }
    if (pathname === '/config.json') {
      res.writeHead(200, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' });
      res.end(req.method === 'HEAD' ? undefined : JSON.stringify(config)); return;
    }
    const file = files.get(pathname);
    if (!file) { send(res, 404, { error: 'Not found.' }); return; }
    try {
      const info = await stat(file);
      res.setHeader('Cache-Control', pathname.startsWith(config.base) ? 'public, max-age=31536000, immutable' : 'no-store');
      res.setHeader('Content-Type', mime[extname(file)] ?? 'application/octet-stream');
      res.setHeader('Content-Length', info.size);
      if (req.method === 'HEAD') { res.end(); return; }
      const stream = createReadStream(file);
      stream.on('error', () => res.destroy()); stream.pipe(res);
    } catch { send(res, 404, { error: 'Asset unavailable.' }); }
  };
}

export async function start() {
  const modelPath = resolve(root, process.env.EMAIL_MODEL_DIR ?? 'models/email-300/minilm');
  const manifest = await readManifest(modelPath);
  const expected = ['forum', 'primary', 'promotion', 'purchase', 'social', 'update'];
  if (manifest.features.kind !== 'minilm' || manifest.task.question.type !== 'choice' || JSON.stringify(manifest.task.labels) !== JSON.stringify(expected)) {
    throw new Error('Use the trained six-category MiniLM email model with EMAIL_MODEL_DIR.');
  }
  const teacher = process.env.EMAIL_TEACHER_MODEL ?? 'typesafe-ai/jev';
  const endpoint = teacherEndpoint({ teacher, teacherUrl: process.env.EMAIL_TEACHER_URL });
  const apiKey = process.env.AI_GATEWAY_API_KEY?.trim();
  const browser = await browserAssets();
  const versions = browser.versions;
  const fingerprint = createHash('sha256').update(JSON.stringify({ manifest, versions, worker: createHash('sha256').update(await readFile(join(root, 'dist/browser-worker.js'))).digest('hex') })).digest('hex').slice(0, 24);
  const base = `/assets/${fingerprint}/`;
  const config = { base, assetsUrl: `${base}sdk/`, labels: manifest.task.labels, questionId: manifest.task.questionId, modelId: manifest.id,
    teacherReady: Boolean(apiKey), teacherModel: teacher, maxTokens: manifest.features.maxTokens,
    policies: { wasm: { calibration: manifest.calibration, thresholdRecommendation: manifest.thresholdRecommendation }, webgpu: null } };
  const files = new Map([
    ['/', join(here, 'index.html')], ['/style.css', join(here, 'style.css')], ['/app.js', join(here, 'app.js')], ['/email.js', join(here, 'email.js')],
    ['/file-help.js', join(here, 'file-help.js')],
    ['/jimothy/browser.js', join(root, 'dist/browser.js')], [`${base}model.json`, join(modelPath, 'model.json')],
  ]);
  for (const file of Object.keys(manifest.files)) files.set(`${base}${file}`, join(modelPath, file));
  for (const [name, source] of browser.files) files.set(`${config.assetsUrl}${name}`, source);
  const port = Number(process.env.EMAIL_DEMO_PORT ?? 4320);
  if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error('EMAIL_DEMO_PORT must be a valid port number.');
  const server = createServer(createHandler({ manifest, config, files, apiKey, teacher, endpoint, port: () => port }));
  server.on('error', error => { console.error(`Cannot start email playground: ${error.message}`); process.exitCode = 1; });
  server.listen(port, '127.0.0.1', () => console.log(`Email playground: http://127.0.0.1:${port}\nJev: ${apiKey ? 'ready' : 'set AI_GATEWAY_API_KEY to enable'}\nPress Ctrl-C to stop.`));
  return server;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  start().catch(error => { console.error(`Cannot start email playground: ${error.message}`); process.exitCode = 1; });
}

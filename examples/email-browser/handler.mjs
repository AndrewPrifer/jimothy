import { createReadStream } from 'node:fs';
import { stat } from 'node:fs/promises';
import { extname } from 'node:path';
import { object, targetFromAnswer } from '../../dist/schema.js';
import { emailState } from './email.js';

const mime = { '.html': 'text/html; charset=utf-8', '.css': 'text/css', '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json', '.wasm': 'application/wasm' };
// End of September 25, 2026 in fixed PST (UTC−08:00).
export const SHARED_KEY_EXPIRES_AT = Date.parse('2026-09-26T08:00:00Z');
const MAX_BODY_BYTES = 400_000;
const send = (res, status, body) => {
  res.writeHead(status, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' });
  res.end(JSON.stringify(body));
};

/** Only this handler holds the key. No email or provider response is written to disk or logs. */
export function createHandler({ manifest, config, files, apiKey, teacher, endpoint, port, deployed = false, fetchImpl = fetch, now = Date.now }) {
  let pending = false, retryAt = 0;
  return async (req, res) => {
    res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
    res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
    res.setHeader('X-Content-Type-Options', 'nosniff');
    res.setHeader('Content-Security-Policy', "default-src 'self'; script-src 'self' 'wasm-unsafe-eval' blob:; style-src 'self'; worker-src 'self' blob:; connect-src 'self' blob:; img-src 'self' data:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'");
    const host = req.headers.host;
    if (!deployed && ![ `127.0.0.1:${port()}`, `localhost:${port()}` ].includes(host)) { send(res, 403, { error: 'Invalid host.' }); return; }
    if (typeof host !== 'string' || !/^[a-zA-Z0-9.:-]+$/.test(host)) { send(res, 403, { error: 'Invalid host.' }); return; }
    const origin = `${deployed ? 'https' : 'http'}://${host}`;
    const pathname = new URL(req.url, origin).pathname;
    if (pathname === '/api/jev') {
      if (req.method !== 'POST') { send(res, 405, { error: 'Use POST.' }); return; }
      // Block other websites from using this local, authenticated proxy.
      if (req.headers.origin !== origin ||
          (req.headers['sec-fetch-site'] && req.headers['sec-fetch-site'] !== 'same-origin')) {
        send(res, 403, { error: 'Use the email playground on this server.' }); return;
      }
      if (req.headers['content-type']?.split(';')[0].trim() !== 'application/json') { send(res, 415, { error: 'Send JSON.' }); return; }
      const visitorKey = req.headers['x-jev-api-key'];
      if (visitorKey !== undefined && (typeof visitorKey !== 'string' || !/^[\x21-\x7e]{1,4096}$/.test(visitorKey))) {
        send(res, 400, { error: 'Enter a valid API key.' }); return;
      }
      const keyForRequest = () => visitorKey ?? (now() < SHARED_KEY_EXPIRES_AT ? apiKey : undefined);
      const requireKey = () => send(res, 401, { code: 'api_key_required', error: 'Enter your Vercel API key to use Jev.' });
      if (!keyForRequest()) { requireKey(); return; }
      if (pending) { send(res, 429, { error: 'A Jev request is already running. Try again shortly.' }); return; }
      if (now() < retryAt) {
        const seconds = Math.ceil((retryAt - now()) / 1000);
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
      const requestKey = keyForRequest();
      if (!requestKey) { requireKey(); return; }
      pending = true;
      const abort = new AbortController();
      res.on('close', () => { if (!res.writableEnded) abort.abort(); });
      try {
        const response = await fetchImpl(endpoint, {
          method: 'POST', redirect: 'error', signal: AbortSignal.any([abort.signal, AbortSignal.timeout(30_000)]),
          headers: { Authorization: `Bearer ${requestKey}`, 'Content-Type': 'application/json' },
          body: JSON.stringify({ model: teacher, state, questions: { [manifest.task.questionId]: manifest.task.question } }),
        });
        if (!response.ok) {
          await response.body?.cancel();
          if (response.status === 429) {
            const after = response.headers.get('retry-after');
            const seconds = after && Number.isFinite(Number(after)) ? Number(after) : (Date.parse(after ?? '') - now()) / 1000;
            const wait = Math.max(1, Math.ceil(Number.isFinite(seconds) ? seconds : 30));
            retryAt = now() + wait * 1000;
            res.setHeader('Retry-After', String(wait));
            send(res, 429, { error: `Jev is rate limited. Try again in ${wait}s.` });
          } else send(res, 502, { error: [401, 403].includes(response.status)
            ? 'Jev rejected the API key.' : `Jev returned HTTP ${response.status}. Try again.` });
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
      res.end(req.method === 'HEAD' ? undefined : JSON.stringify({ ...config,
        teacherReady: Boolean(apiKey) && now() < SHARED_KEY_EXPIRES_AT,
        sharedKeyExpiresAt: SHARED_KEY_EXPIRES_AT, serverTime: now(),
      })); return;
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


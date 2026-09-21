import test from 'node:test';
import assert from 'node:assert/strict';
import { createServer, request } from 'node:http';
import { readFile, readdir } from 'node:fs/promises';
import { createHash } from 'node:crypto';

test('built functions support HTTPS origins, dynamic config, and visitor credentials', async () => {
  // No shared credential and no external network calls in this test.
  delete process.env.AI_GATEWAY_API_KEY;
  const previousFetch = globalThis.fetch;
  let calls = 0;
  globalThis.fetch = async (_url, init) => {
    calls++;
    assert.equal(init.headers.Authorization, 'Bearer visitor-test-key');
    assert.deepEqual(Object.keys(JSON.parse(init.body).state), ['body', 'from', 'subject']);
    return Response.json({ answers: { category: { type: 'choice', choice: 'purchase',
      probabilities: { forum: 0, primary: 0, promotion: 0, purchase: 1, social: 0, update: 0 } } } });
  };
  const { default: api } = await import('../.vercel/output/functions/api/jev.func/index.mjs');
  const { default: config } = await import('../.vercel/output/functions/config.json.func/index.mjs');
  const server = createServer((req, res) => req.url === '/config.json' ? config(req, res) : api(req, res));
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const send = (path, origin, key) => new Promise((resolve, reject) => {
    const req = request({ agent: false, hostname: '127.0.0.1', port: server.address().port, path,
      method: path === '/config.json' ? 'GET' : 'POST',
      headers: { Host: 'jimothy.example', Origin: origin, 'Content-Type': 'application/json',
        ...(key ? { 'X-Jev-Api-Key': key } : {}) } }, res => {
      let text = ''; res.on('data', chunk => text += chunk);
      res.on('end', () => resolve({ status: res.statusCode, headers: res.headers, body: JSON.parse(text) }));
    });
    req.on('error', reject);
    req.end(path === '/config.json' ? undefined : JSON.stringify({ from: 'shop@example.com', subject: 'Receipt', body: 'Payment received.' }));
  });
  try {
    const settings = await send('/config.json', 'https://jimothy.example');
    assert.equal(settings.status, 200);
    assert.equal(settings.body.teacherReady, false);
    assert.equal(settings.headers['cache-control'], 'no-store');
    assert.equal((await send('/api/jev', 'http://jimothy.example', 'visitor-test-key')).status, 403);
    assert.equal((await send('/api/jev', 'https://other.example', 'visitor-test-key')).status, 403);
    assert.equal((await send('/api/jev', 'https://jimothy.example')).status, 401);
    assert.equal((await send('/api/jev', 'https://jimothy.example', 'visitor-test-key')).status, 200);
    assert.equal(calls, 1);
  } finally {
    globalThis.fetch = previousFetch;
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
  }
});

test('static deployment contains the intact model and browser isolation headers', async () => {
  const output = new URL('../.vercel/output/', import.meta.url);
  const routing = JSON.parse(await readFile(new URL('config.json', output)));
  assert.equal(routing.version, 3);
  assert.equal(routing.routes[0].headers['Cross-Origin-Embedder-Policy'], 'require-corp');
  const [fingerprint] = await readdir(new URL('static/assets/', output));
  const base = new URL('static/assets/' + fingerprint + '/', output);
  const manifest = JSON.parse(await readFile(new URL('model.json', base)));
  for (const [file, hash] of Object.entries(manifest.files)) {
    assert.equal(createHash('sha256').update(await readFile(new URL(file, base))).digest('hex'), hash);
  }
  await readFile(new URL('sdk/browser-worker.js', base));
  await readFile(new URL('sdk/runtime/ort-wasm-simd-threaded.wasm', base));
  const source = await readFile(new URL('functions/api/jev.func/index.mjs', output), 'utf8');
  assert.ok(!source.includes('@huggingface/transformers'));
});

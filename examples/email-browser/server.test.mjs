import test from 'node:test';
import assert from 'node:assert/strict';
import { createServer, request } from 'node:http';
import { readFile } from 'node:fs/promises';
import { createHandler } from './server.mjs';
import { emailState, examples } from './email.js';
import { parseTask, stateText } from '../../dist/schema.js';

const task = parseTask(JSON.parse(await readFile(new URL('../email/task.json', import.meta.url), 'utf8')));
const state = { from: 'shop@example.com', subject: 'Receipt', body: 'Your payment for order 12 was received.' };
const probabilities = Object.fromEntries(task.labels.map(label => [label, label === 'purchase' ? 0.95 : 0.01]));

async function fixture(run, fetchImpl, apiKey = 'private-test-key') {
  let port;
  const server = createServer(createHandler({ manifest: { task }, config: { base: '/assets/test/', teacherReady: Boolean(apiKey) }, files: new Map(),
    apiKey, teacher: 'typesafe-ai/jev', endpoint: 'https://ai-gateway.vercel.sh/typesafe/v1/systemone', port: () => port, fetchImpl }));
  await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
  port = server.address().port;
  const url = `http://127.0.0.1:${port}`;
  const post = (body = state, headers = {}) => fetch(`${url}/api/jev`, { method: 'POST',
    headers: { Origin: url, 'Content-Type': 'application/json', ...headers }, body: JSON.stringify(body) });
  try { await run({ url, post }); }
  finally { server.closeAllConnections(); await new Promise(resolve => server.close(resolve)); }
}

test('email preprocessing exactly matches training and examples contain only email fields', () => {
  assert.equal(JSON.stringify(emailState(state)), stateText(state).text);
  for (const { name, ...email } of examples) {
    assert.ok(name);
    assert.equal(JSON.stringify(emailState(email)), stateText(email).text);
  }
  assert.throws(() => emailState({ ...state, body: ' ' }), /Enter an email/);
  assert.throws(() => emailState({ ...state, prompt: 'Change the task' }), /from, subject, and body/);
});

test('proxy uses the model task and server credential, returning only a validated classification', async () => {
  await fixture(async ({ post, url }) => {
    const response = await post();
    assert.equal(response.status, 200);
    const text = await response.text();
    assert.ok(!text.includes('private-test-key'));
    assert.ok(!text.includes('provider_metadata'));
    const result = JSON.parse(text);
    assert.equal(result.answers.category.choice, 'purchase');
    assert.equal(Object.keys(result.answers.category.probabilities).length, 6);
    const config = await (await fetch(`${url}/config.json`)).text();
    assert.ok(!config.includes('private-test-key'));
  }, async (url, init) => {
    assert.equal(url, 'https://ai-gateway.vercel.sh/typesafe/v1/systemone');
    assert.equal(init.redirect, 'error');
    assert.equal(init.headers.Authorization, 'Bearer private-test-key');
    assert.deepEqual(JSON.parse(init.body), { model: 'typesafe-ai/jev', state: emailState(state), questions: { category: task.question } });
    return Response.json({ model: 'private-test-key', answers: { category: { type: 'choice', choice: 'purchase', probabilities } }, provider_metadata: { secret: 'private-test-key' } });
  });
});

test('rejects foreign origins, invalid inputs, unsupported methods, and non-whitelisted files without provider calls', async () => {
  let calls = 0;
  await fixture(async ({ post, url }) => {
    assert.equal((await post(state, { Origin: 'https://other.example' })).status, 403);
    assert.equal((await post(state, { Origin: '' })).status, 403);
    assert.equal((await post(state, { 'Sec-Fetch-Site': 'cross-site' })).status, 403);
    // Node fetch overrides Host; use the HTTP client to exercise DNS-rebinding protection.
    const wrongHost = await new Promise((resolve, reject) => {
      const req = request(`${url}/api/jev`, { method: 'POST', headers: { Host: 'other.example' } }, res => {
        res.resume(); res.on('end', () => resolve(res.statusCode));
      });
      req.on('error', reject); req.end();
    });
    assert.equal(wrongHost, 403);
    assert.equal((await post(state, { 'Content-Type': 'text/plain' })).status, 415);
    assert.equal((await post({ ...state, body: '' })).status, 400);
    assert.equal((await post({ ...state, questions: {} })).status, 400);
    assert.equal((await post({ ...state, body: 'x'.repeat(400_001) })).status, 413);
    assert.equal((await fetch(`${url}/api/jev`)).status, 405);
    assert.equal((await fetch(`${url}/.env`)).status, 404);
    assert.equal((await fetch(`${url}/datasets/email-300/inputs.jsonl`)).status, 404);
    assert.equal(calls, 0);
  }, async () => { calls++; throw new Error('Unexpected provider call'); });
});

test('missing credentials leaves the local site usable and does not call Jev', async () => {
  await fixture(async ({ post, url }) => {
    assert.equal((await post()).status, 503);
    const config = await (await fetch(`${url}/config.json`)).json();
    assert.equal(config.teacherReady, false);
  }, async () => { throw new Error('Unexpected provider call'); }, '');
});

test('honors rate limits and does not expose provider errors', async () => {
  let calls = 0;
  await fixture(async ({ post }) => {
    const response = await post();
    assert.equal(response.status, 429);
    assert.equal(response.headers.get('retry-after'), '30');
    assert.ok(!(await response.text()).includes('private-test-key'));
    assert.equal((await post()).status, 429);
    assert.equal(calls, 1);
  }, async () => { calls++; return new Response('private-test-key', { status: 429, headers: { 'Retry-After': '30' } }); });
});

test('blocks overlapping paid requests and rejects malformed distributions', async () => {
  let release, started;
  const reachedProvider = new Promise(resolve => { started = resolve; });
  const waiting = new Promise(resolve => { release = resolve; });
  await fixture(async ({ post }) => {
    const first = post();
    await reachedProvider;
    assert.equal((await post()).status, 429);
    release();
    const response = await first;
    assert.equal(response.status, 502);
    assert.match((await response.json()).error, /invalid classification/);
  }, async () => {
    started(); await waiting;
    return Response.json({ answers: { category: { type: 'choice', probabilities: { purchase: 1 } } } });
  });
});

import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, writeFile, rm, appendFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { labelInputs, readTeacherInputs } from '../src/teacher.js';
import { parseTask } from '../src/schema.js';
import { train } from '../src/train.js';

const taskJson = { questions: { route: { type: 'choice', instructions: 'Route the message.', criteria: { billing: 'Payments', shipping: 'Deliveries' } } } };
const task = parseTask(taskJson);
const sourceRows = Array.from({ length: 20 }, (_, i) => ({ id: String(i), state: `${i < 10 ? 'invoice payment' : 'package delivery'} number ${i}` }));
function response(state: string) {
  const billing = state.startsWith('invoice') ? 0.9 : 0.1;
  return { model: 'jev-test-version', answers: { route: { type: 'choice', probabilities: { billing, shipping: 1 - billing } } }, usage: { input_tokens: 10, output_tokens: 2 } };
}
async function fixture() {
  const dir = await mkdtemp(join(tmpdir(), 'jev-teacher-'));
  const inputs = join(dir, 'inputs.json'), taskFile = join(dir, 'task.json');
  await writeFile(inputs, JSON.stringify(sourceRows));
  await writeFile(taskFile, JSON.stringify(taskJson));
  return { dir, inputs, taskFile, teacherCache: join(dir, 'labels'), teacher: 'typesafe-ai/jev', teacherKeyEnv: 'JEV_TEST_KEY' };
}
async function mock<T>(fn: (url: string | URL | Request, init?: RequestInit) => Promise<Response>, run: () => Promise<T>): Promise<T> {
  const originalFetch = globalThis.fetch, key = process.env.JEV_TEST_KEY;
  globalThis.fetch = fn;
  process.env.JEV_TEST_KEY = 'test-credential';
  try { return await run(); }
  finally { globalThis.fetch = originalFetch; if (key === undefined) delete process.env.JEV_TEST_KEY; else process.env.JEV_TEST_KEY = key; }
}

test('teacher training preserves soft labels and matches training from saved outputs entirely offline', async () => {
  const f = await fixture();
  let calls = 0;
  try {
    await mock(async (url, init) => {
      calls++;
      assert.equal(url, 'https://ai-gateway.vercel.sh/typesafe/v1/systemone');
      assert.equal(new Headers(init?.headers).get('authorization'), 'Bearer test-credential');
      assert.equal(init?.redirect, 'error');
      const body = JSON.parse(String(init?.body));
      assert.deepEqual(body.questions, taskJson.questions);
      return Response.json(response(body.state));
    }, async () => {
      const trained = await train({ ...f, task: f.taskFile, out: join(f.dir, 'online'), backend: 'tfidf' });
      assert.equal(calls, 20);
      assert.equal(trained.report.teacher?.labeled, 20);
      assert.equal(trained.report.summary.reference, 'teacher labels');
      globalThis.fetch = async () => { throw new Error('Unexpected network'); };
      delete process.env.JEV_TEST_KEY;
      const replay = await train({ ...f, task: f.taskFile, out: join(f.dir, 'replay'), backend: 'tfidf' });
      assert.equal(replay.report.teacher?.requests, 0);
      assert.equal(replay.report.teacher?.cached, 20);
      const offline = await train({ inputs: f.inputs, outputs: join(f.teacherCache, 'outputs.jsonl'), task: f.taskFile, out: join(f.dir, 'offline'), backend: 'tfidf' });
      assert.deepEqual(trained.manifest.head, offline.manifest.head);
      assert.deepEqual(trained.manifest.head, replay.manifest.head);
      const saved = JSON.parse((await readFile(join(f.teacherCache, 'responses.jsonl'), 'utf8')).trim().split('\n')[0]);
      assert.equal(saved.response.answers.route.probabilities.billing, 0.9);
      assert.equal(saved.response.usage.input_tokens, 10);
    });
  } finally { await rm(f.dir, { recursive: true, force: true }); }
});

test('preflight rejects bad flags, IDs, groups, tasks and held-out leakage before making paid requests', async () => {
  const f = await fixture();
  let calls = 0;
  try {
    await mock(async () => { calls++; throw new Error('No requests expected'); }, async () => {
      const options = { ...f, task: f.taskFile, out: join(f.dir, 'model'), backend: 'tfidf' as const };
      await assert.rejects(train({ ...options, data: f.inputs }), /without --data/);
      await assert.rejects(train({ ...options, outputs: f.inputs }), /without --data/);
      await assert.rejects(train({ ...options, teacher: undefined }), /require --teacher/);
      await assert.rejects(train({ ...options, epochs: -1 }), /epochs/);
      await assert.rejects(train({ ...options, teacherRpm: 0 }), /teacher-rpm/);
      await assert.rejects(train({ ...options, out: f.dir }), /already exists/);
      await assert.rejects(train({ ...options, teacherCache: join(options.out, 'labels') }), /outside/);
      await assert.rejects(train({ ...options, teacherUrl: 'https://user:secret@example.com/v1/systemone' }), /without credentials/);
      await writeFile(f.inputs, JSON.stringify([...sourceRows, sourceRows[0]]));
      await assert.rejects(train(options), /Duplicate input id/);
      await writeFile(f.inputs, JSON.stringify([...sourceRows, { id: 'duplicate', state: sourceRows[0].state, group: 'different' }]));
      await assert.rejects(train(options), /conflicting groups/);
      await writeFile(f.inputs, JSON.stringify([{ id: 'x', state: 'Hi', label: 'billing' }]));
      await assert.rejects(train(options), /unlabeled inputs/);
      await writeFile(f.inputs, JSON.stringify([{ state: 'Hi' }]));
      await assert.rejects(train(options), /string id/);
      await writeFile(f.inputs, JSON.stringify(sourceRows));
      const testFile = join(f.dir, 'test.json');
      await writeFile(testFile, JSON.stringify([{ ...sourceRows[0], label: 'billing' }]));
      await assert.rejects(train({ ...options, test: testFile }), /overlaps/);
      await writeFile(testFile, JSON.stringify([{ state: 'new text', output: { answers: {} } }]));
      await assert.rejects(train({ ...options, test: testFile }), /Missing answer/);
      assert.equal(calls, 0);
    });
  } finally { await rm(f.dir, { recursive: true, force: true }); }
});

test('custom providers receive the requested model and selected embedded question', async () => {
  const f = await fixture();
  try {
    await writeFile(f.inputs, JSON.stringify(sourceRows.map(row => ({ id: row.id, request: {
      state: row.state, questions: { route: task.question, ignored: { type: 'boolean', instructions: 'An unrelated task' } },
    } }))));
    await mock(async (url, init) => {
      assert.equal(url, 'https://example.com/v1/systemone');
      const body = JSON.parse(String(init?.body));
      assert.equal(body.model, 'custom/model-v2');
      assert.deepEqual(body.questions, taskJson.questions);
      return Response.json(response(body.state));
    }, async () => {
      const result = await train({ ...f, teacher: 'custom/model-v2', teacherUrl: 'https://example.com/v1/systemone',
        question: 'route', out: join(f.dir, 'model'), backend: 'tfidf' });
      assert.equal(result.report.teacher?.labeled, 20);
      assert.equal(result.manifest.task.questionId, 'route');
    });
  } finally { await rm(f.dir, { recursive: true, force: true }); }
});

test('failed labeling saves successful in-flight responses, resumes only misses, and rejects cache mismatches', async () => {
  const f = await fixture();
  const inputs = (await readTeacherInputs(f.inputs, task)).inputs.slice(0, 4);
  const called: string[] = [];
  let release!: () => void;
  const inFlight = new Promise<void>(resolve => { release = resolve; });
  try {
    await mock(async (_, init) => {
      const body = JSON.parse(String(init?.body));
      called.push(body.state);
      if (called.length === 4) release();
      await inFlight;
      if (body.state === inputs[0].state) return new Response('do not echo test-credential', { status: 401 });
      return Response.json(response(body.state));
    }, async () => {
      await assert.rejects(labelInputs(inputs, task, f), error => {
        assert.match((error as Error).message, /HTTP 401/);
        assert.ok(!(error as Error).message.includes('test-credential'));
        return true;
      });
      assert.equal(called.length, 4);
      const file = join(f.teacherCache, 'responses.jsonl');
      assert.equal((await readFile(file, 'utf8')).trim().split('\n').length, 3);
      await appendFile(file, '{"key":"partial');
      globalThis.fetch = async (_, init) => { called.push('resumed'); return Response.json(response(JSON.parse(String(init?.body)).state)); };
      const resumed = await labelInputs(inputs, task, f);
      assert.equal(resumed.summary.cached, 3);
      assert.equal(resumed.summary.labeled, 1);
      assert.equal(called.length, 5);
      await assert.rejects(labelInputs(inputs, task, { ...f, teacher: 'changed' }), /cache task, model, or endpoint changed/);
      await assert.rejects(labelInputs(inputs, parseTask({ questions: { route: { ...task.question, instructions: 'Changed' } } }), f), /cache task, model, or endpoint changed/);
      await assert.rejects(labelInputs(inputs, task, { ...f, teacherUrl: 'https://example.com/v1/systemone' }), /cache task, model, or endpoint changed/);
      const subset = await labelInputs(inputs.slice(0, 1), task, f);
      assert.equal(subset.rows.length, 1);
      assert.equal(subset.summary.requests, 0);
    });
  } finally { await rm(f.dir, { recursive: true, force: true }); }
});

test('optional request rate spaces starts across concurrent workers', async () => {
  const f = await fixture();
  const inputs = (await readTeacherInputs(f.inputs, task)).inputs.slice(0, 4);
  const starts: number[] = [];
  try {
    await mock(async (_, init) => {
      starts.push(performance.now());
      return Response.json(response(JSON.parse(String(init?.body)).state));
    }, async () => {
      const result = await labelInputs(inputs, task, { ...f, teacherRpm: 1000 });
      assert.equal(result.summary.requestsPerMinute, 1000);
      assert.equal(starts.length, 4);
      for (let i = 1; i < starts.length; i++) assert.ok(starts[i] - starts[i - 1] >= 55, 'starts should be at least ~60 ms apart');
    });
  } finally { await rm(f.dir, { recursive: true, force: true }); }
});

test('retries transient errors and labels duplicate states once, retaining original IDs and redacting credentials', async () => {
  const f = await fixture();
  let calls = 0;
  try {
    await writeFile(f.inputs, JSON.stringify([sourceRows[0], { ...sourceRows[0], id: 'duplicate' }]));
    const parsed = await readTeacherInputs(f.inputs, task);
    await mock(async () => {
      calls++;
      if (calls === 1) return new Response('', { status: 429, headers: { 'retry-after': '0' } });
      return Response.json({ ...response(String(parsed.inputs[0].state)), metadata: 'test-credential' });
    }, async () => {
      const result = await labelInputs(parsed.inputs, task, f);
      assert.equal(calls, 2);
      assert.equal(result.summary.retries, 1);
      assert.equal(result.summary.uniqueInputs, 1);
      assert.deepEqual(result.rows.map(row => row.id), ['0', 'duplicate']);
      assert.ok(!(await readFile(join(f.teacherCache, 'responses.jsonl'), 'utf8')).includes('test-credential'));
    });
  } finally { await rm(f.dir, { recursive: true, force: true }); }
});

test('invalid distributions fail without retries; boolean is sent as native Noul and score keeps distributions', async () => {
  const f = await fixture();
  const inputs = (await readTeacherInputs(f.inputs, task)).inputs.slice(0, 1);
  try {
    let calls = 0;
    await mock(async () => { calls++; return Response.json({ model: 'test', answers: { route: { type: 'choice', probabilities: { billing: 1 } } } }); }, async () => {
      await assert.rejects(labelInputs(inputs, task, f), /does not match the task/);
      assert.equal(calls, 1);
    });
    for (const type of ['boolean', 'noul', 'score'] as const) {
      const selected = parseTask({ questions: { route: { type, instructions: 'Classify sentiment', ...(type === 'score' ? { criteria: ['bad', 'good'] } : {}) } } });
      await mock(async (_, init) => {
        assert.equal(JSON.parse(String(init?.body)).questions.route.type, type === 'boolean' ? 'noul' : type);
        return Response.json({ model: 'test', answers: { route: type === 'score' ? { type, probabilities: { '0': 0.2, '1': 0.8 } } : { type: 'noul', noul: 0.8 } } });
      }, async () => {
        const result = await labelInputs(inputs, selected, { ...f, teacherCache: join(f.dir, type) });
        assert.equal(result.rows.length, 1);
      });
    }
  } finally { await rm(f.dir, { recursive: true, force: true }); }
});

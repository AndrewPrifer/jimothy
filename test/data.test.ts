import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { assertDisjoint, joinFiles, parseDataset, splitExamples } from '../src/data.js';
import { canonical, parseTask, stateText, targetFromAnswer } from '../src/schema.js';

const questions = { route: { type: 'choice', instructions: 'Route this ticket', criteria: { support: 'bugs', billing: 'payments' } } };
const task = parseTask({ questions });

test('imports native request/response logs and preserves soft targets by label name', () => {
  const result = parseDataset([{ request: { state: { b: 2, a: 'hello' }, questions }, response: {
    model: 'permitted-teacher-v1', answers: { route: { type: 'choice', choice: 'billing', probabilities: { support: 0.2, billing: 0.8 }, confidence: 0.6 } },
  } }]);
  assert.deepEqual(result.task.labels, ['billing', 'support']);
  assert.deepEqual(result.examples[0].target, [0.8, 0.2]);
  assert.equal(result.examples[0].text, '{"a":"hello","b":2}');
  assert.equal(result.examples[0].teacherModel, 'permitted-teacher-v1');
});

test('human corrections override training labels without losing the teacher reference', () => {
  const result = parseDataset([{ state: 'a bug', label: 'support', answers: { route: { choice: 'billing' } } }], task);
  assert.deepEqual(result.examples[0].target, [0, 1]);
  assert.deepEqual(result.examples[0].teacher, [1, 0]);
  assert.equal(result.examples[0].humanLabel, 1);
});

test('normalizes Noul and Vercel Boolean probabilities, and preserves ordered Score distributions', () => {
  const binary = parseTask({ flagged: { type: 'noul', instructions: 'Is it broken?' } });
  assert.deepEqual(targetFromAnswer({ type: 'boolean', probability: 0.75 }, binary), [0.25, 0.75]);
  const score = parseTask({ quality: { type: 'score', instructions: 'Rate quality', criteria: ['poor', 'fair', 'good'] } });
  assert.deepEqual(targetFromAnswer({ probabilities: { '2': 0.3, '0': 0.2, '1': 0.5 } }, score), [0.2, 0.5, 0.3]);
  assert.throws(() => targetFromAnswer({ score: 1.2 }, score), /Score means alone/);
});

test('rejects malformed probabilities, labels, and task drift', () => {
  assert.throws(() => targetFromAnswer({ probabilities: { billing: 0.9 } }, task), /exactly/);
  assert.throws(() => targetFromAnswer({ probabilities: { billing: 0.9, support: 0.5 } }, task), /sum to 1/);
  assert.throws(() => targetFromAnswer({ probabilities: { billing: -0.1, support: 1.1 } }, task), /between 0 and 1/);
  assert.throws(() => targetFromAnswer({ choice: 'other' }, task), /Unknown label/);
  assert.throws(() => parseDataset([{ input: { state: 'hi', questions: { route: { ...questions.route, instructions: 'New policy' } } }, output: 'billing' }], task), /changed/);
  const normalized = targetFromAnswer({ probabilities: { billing: 0.5, support: 0.499 } }, task);
  assert.ok(Math.abs(normalized.reduce((a,b) => a+b, 0) - 1) < 1e-12);
});

test('deduplicates identical inputs and rejects conflicting targets', () => {
  const row = { state: 'same text', output: 'billing' };
  assert.equal(parseDataset([row, row], task).duplicatesRemoved, 1);
  assert.throws(() => parseDataset([row, { state: 'same text', output: 'support' }], task), /conflicting/);
  assert.throws(() => parseDataset([{ ...row, id: 'x' }, { state: 'other text', output: 'support', id: 'x' }], task), /Duplicate id/);
});

test('separate datasets join by id, with no positional fallback or orphan outputs', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'jev-join-'));
  try {
    const input = join(dir, 'in.json'), output = join(dir, 'out.json');
    await writeFile(input, JSON.stringify([{ id: 'a', state: 'first' }, { id: 'b', state: 'second' }]));
    await writeFile(output, JSON.stringify([{ id: 'b', answers: { route: { choice: 'support' } } }, { id: 'a', answers: { route: { choice: 'billing' } } }]));
    const dataset = parseDataset(await joinFiles(input, output), task);
    assert.deepEqual(dataset.examples.map(e => e.target), [[1, 0], [0, 1]]);
    await writeFile(output, JSON.stringify([{ id: 'a', answers: {} }]));
    await assert.rejects(joinFiles(input, output), /Missing output/);
    await writeFile(input, JSON.stringify([{ state: 'missing id' }]));
    await assert.rejects(joinFiles(input, output), /Missing output/);
  } finally { await rm(dir, { recursive: true, force: true }); }
});

test('group split is deterministic and prevents input/group leakage', () => {
  const examples = parseDataset(Array.from({ length: 40 }, (_, i) => ({ state: `example ${i}`, group: `g${Math.floor(i/2)}`, label: i < 20 ? 'billing' : 'support' })), task).examples;
  const first = splitExamples(examples, 42), second = splitExamples(examples, 42);
  assert.deepEqual(first, second);
  assertDisjoint(first.train, first.validation, 'Validation');
  assert.throws(() => assertDisjoint(first.train, [first.train[0]], 'Validation'), /overlaps/);
  const related = { ...first.train[0], text: 'new wording' };
  assert.throws(() => assertDisjoint(first.train, [related], 'Validation'), /overlaps/);
});

test('requires explicit selection for multi-question tasks', () => {
  const multi = { ...questions, other: { type: 'boolean', instructions: 'Does it mention money?' } };
  assert.throws(() => parseTask(multi), /Select a question/);
  assert.equal(parseTask(multi, 'route').questionId, 'route');
});

test('structured inputs preserve array order, reserved JSON keys, and reject non-finite values', () => {
  assert.equal(stateText({ b: [2, 1], a: 'x' }).text, stateText({ a: 'x', b: [2, 1] }).text);
  assert.notEqual(stateText({ b: [2, 1] }).text, stateText({ b: [1, 2] }).text);
  assert.equal(canonical(JSON.parse('{"__proto__":"data"}')), '{"__proto__":"data"}');
  assert.throws(() => stateText({ value: Infinity }), /finite JSON/);
  assert.throws(() => stateText('  '), /empty/);
});

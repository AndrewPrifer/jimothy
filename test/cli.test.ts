import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const cli = (...args: string[]) => spawnSync(process.execPath, ['--import', 'tsx', 'src/cli.ts', ...args], { encoding: 'utf8' });

test('CLI validates, trains, predicts, and evaluates with machine-readable stdout', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'jev-cli-'));
  try {
    const validation = cli('validate', '--task', 'examples/task.json', '--data', 'examples/training.jsonl');
    assert.equal(validation.status, 0, validation.stderr);
    assert.equal(JSON.parse(validation.stdout).uniqueExamples, 36);
    const out = join(dir, 'model');
    const training = cli('train', '--task', 'examples/task.json', '--data', 'examples/training.jsonl', '--backend', 'tfidf', '--out', out);
    assert.equal(training.status, 0, training.stderr);
    assert.equal(JSON.parse(training.stdout).backend, 'tfidf');
    assert.equal(JSON.parse(training.stdout).summary.reference, 'teacher labels');
    assert.equal(JSON.parse(training.stdout).thresholdRecommendation.status, 'insufficient_data');
    assert.equal(JSON.parse(training.stdout).test, undefined);
    assert.match(training.stderr, /Exported/);
    const prediction = cli('predict', '--model', out, '--text', 'Where is my parcel delivery?');
    assert.equal(prediction.status, 0, prediction.stderr);
    assert.equal(JSON.parse(prediction.stdout).answers.route.choice, 'shipping');
    assert.deepEqual(Object.keys(JSON.parse(prediction.stdout)).sort(), ['answers', 'model']);
    const inspection = cli('inspect', '--model', out);
    assert.equal(inspection.status, 0, inspection.stderr);
    assert.deepEqual(JSON.parse(inspection.stdout).thresholdRecommendation, JSON.parse(training.stdout).thresholdRecommendation);
    assert.equal(JSON.parse(inspection.stdout).thresholdRecommendation.threshold, null);
    assert.equal(JSON.parse(training.stdout).summary.coverage, null);
    const batch = cli('predict', '--model', out, '--data', 'examples/test.jsonl');
    assert.equal(batch.status, 0, batch.stderr);
    assert.equal(batch.stdout.trim().split('\n').length, 9);
    assert.equal(JSON.parse(batch.stdout.trim().split('\n')[0]).id, 'gold-0');
    const evaluation = cli('evaluate', '--model', out, '--data', 'examples/test.jsonl');
    assert.equal(evaluation.status, 0, evaluation.stderr);
    assert.equal(JSON.parse(evaluation.stdout).summary.examples, 9);
  } finally { await rm(dir, { recursive: true, force: true }); }
});

test('CLI produces actionable errors and rejects unknown or inapplicable options', () => {
  assert.equal(cli('--help').status, 0);
  assert.equal(cli('--version').stdout.trim(), '0.1.0');
  const invalid = cli('train', '--out', '/tmp/unused', '--epochs', 'wat');
  assert.equal(invalid.status, 1);
  assert.match(invalid.stderr, /finite number/);
  assert.equal(invalid.stdout, '');
  assert.equal(cli('predict', '--backend', 'tfidf').status, 1);
  assert.equal(cli('train', '--unknown').status, 1);
  for (const command of ['train', 'predict', 'evaluate']) assert.match(cli(command, '--threshold', '0.8').stderr, /Unknown option/);
});

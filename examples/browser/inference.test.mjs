import test from 'node:test';
import assert from 'node:assert/strict';
import { parseInputs, makePrediction } from './inference.js';

test('single inputs preserve internal newlines and batches exclude blank lines', () => {
  assert.deepEqual(parseInputs(' First line\nsecond line ', 'single'), ['First line\nsecond line']);
  assert.deepEqual(parseInputs(' one\r\n\n two \n', 'batch'), ['one', 'two']);
  assert.throws(() => parseInputs('  ', 'single'), /Enter a message/);
  assert.throws(() => parseInputs('one\n'.repeat(51), 'batch'), /50 messages/);
  assert.throws(() => parseInputs('x'.repeat(100001), 'single'), /shorter/);
});

test('browser inference returns probabilities and Jev-compatible answers without decisions', () => {
  const task = { questionId: 'intent', labels: ['billing', 'card'] };
  for (const probabilities of [[0.88, 0.12], [0.5, 0.5], [1, 0]]) {
    const result = makePrediction({ answer: { type: 'choice', choice: 'billing', probabilities: { billing: probabilities[0], card: probabilities[1] } }, maxProbability: probabilities[0] }, { task, id: 'local' });
    assert.equal(result.maxProbability, probabilities[0]);
    assert.equal('accepted' in result, false);
    assert.equal('reason' in result, false);
    assert.deepEqual(result.response, { model: 'local', answers: { intent: {
      type: 'choice', choice: 'billing', probabilities: { billing: probabilities[0], card: probabilities[1] },
    } } });
  }
});

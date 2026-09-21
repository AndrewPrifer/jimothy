import test from 'node:test';
import assert from 'node:assert/strict';
import { validateLogits, evaluatePolicy, verifyBrowserDecisions } from './analysis.mjs';

const examples = [{ id: 'a', text: 'a', target: [1, 0], humanLabel: 0 }, { id: 'b', text: 'b', target: [0, 1], humanLabel: 1 }];
const policy = { calibration: { temperature: 1 }, threshold: 0.8, acceptance: { status: 'ready' } };

test('browser logits cannot omit, reorder or corrupt held-out examples', () => {
  const rows = [{ id: 'a', logits: [2, 0] }, { id: 'b', logits: [0, 2] }];
  assert.deepEqual(validateLogits(rows, examples, 2), [[2, 0], [0, 2]]);
  assert.throws(() => validateLogits(rows.slice(1), examples, 2), /every example/);
  assert.throws(() => validateLogits([...rows].reverse(), examples, 2), /ID/);
  assert.throws(() => validateLogits([{ id: 'a', logits: [2, NaN] }, rows[1]], examples, 2), /Invalid logits/);
  assert.throws(() => validateLogits([{ id: 'a', logits: [2] }, rows[1]], examples, 2), /Invalid logits/);
});

test('verification detects stale browser temperature or acceptance policy', () => {
  const logits = [[2, 0], [0, 2]];
  const expected = evaluatePolicy(logits, examples, ['left', 'right'], policy);
  const stale = evaluatePolicy(logits, examples, ['left', 'right'], { ...policy, calibration: { temperature: 2 } });
  assert.equal(expected.summary.accepted, 2);
  assert.equal(stale.summary.accepted, 0);
  verifyBrowserDecisions(expected.decisions, expected.decisions);
  assert.throws(() => verifyBrowserDecisions(stale.decisions, expected.decisions), /saved policy/);
  assert.throws(() => verifyBrowserDecisions([{ ...expected.decisions[0], maxProbability: NaN }, expected.decisions[1]], expected.decisions), /saved policy/);
});

test('evaluation preserves a frozen policy and explicit abstention', () => {
  const frozen = Object.freeze({ calibration: Object.freeze({ temperature: 1 }), threshold: 1,
    acceptance: Object.freeze({ status: 'target_not_met' }) });
  const result = evaluatePolicy([[1000, 0], [0, 1000]], examples, ['left', 'right'], frozen);
  assert.equal(result.summary.accuracy, 1);
  assert.equal(result.summary.accepted, 0);
  assert.equal(result.summary.acceptedAccuracy, null);
  assert.equal(frozen.threshold, 1);
});

import { argmax } from './data.js';
import { softmax } from './linear.js';
import type { Calibration, ThresholdRecommendation } from './types.js';

export const MIN_CALIBRATION_EXAMPLES = 30;
export const THRESHOLDS = [0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.925, 0.95, 0.975, 0.99, 0.995, 1];

export function hardLogLoss(logits: number[][], labels: number[], temperature: number): number {
  return logits.reduce((sum, row, i) => {
    const max = Math.max(...row);
    // Stable log-softmax: do not clip probabilities in the calibration objective.
    return sum + (max - row[labels[i]]) / temperature + Math.log(row.reduce((s, z) => s + Math.exp((z - max) / temperature), 0));
  }, 0) / logits.length;
}

export function fitTemperature(logits: number[][], labels: number[]) {
  if (logits.length !== labels.length) throw new Error('Calibration inputs and labels must match.');
  const base: Calibration = { method: 'temperature', temperature: 1, status: 'insufficient_data' };
  if (logits.length < MIN_CALIBRATION_EXAMPLES) return { ...base, examples: logits.length, lossBefore: null, lossAfter: null };
  // Hard reference labels calibrate correctness of the winning answer, including
  // when the reference is the winning label in a teacher's soft distribution.
  const objective = (logT: number) => hardLogLoss(logits, labels, Math.exp(logT));
  let low = Math.log(0.05), high = Math.log(20);
  for (let i = 0; i < 80; i++) {
    const a = low + (high - low) / 3, b = high - (high - low) / 3;
    if (objective(a) <= objective(b)) high = b; else low = a;
  }
  const lossBefore = objective(0);
  const candidate = Math.exp((low + high) / 2);
  const temperature = hardLogLoss(logits, labels, candidate) < lossBefore ? candidate : 1;
  return { method: 'temperature' as const, temperature, status: 'fitted' as const, examples: logits.length,
    lossBefore, lossAfter: hardLogLoss(logits, labels, temperature) };
}

/** One-sided exact binomial (Clopper–Pearson) lower confidence bound. */
export function accuracyLowerBound(correct: number, count: number, alpha = 0.05): number {
  if (!Number.isInteger(count) || !Number.isInteger(correct) || count < 0 || correct < 0 || correct > count || !(alpha > 0 && alpha < 1)) {
    throw new Error('Invalid binomial counts or tail probability.');
  }
  if (correct === 0) return 0;
  if (correct === count) return alpha ** (1 / count);
  // P(X >= correct) under p increases with p. Invert it at alpha.
  let logCombination = 0;
  for (let j = 1; j <= Math.min(correct, count - correct); j++) logCombination += Math.log(count - j + 1) - Math.log(j);
  let low = 0, high = correct / count;
  for (let step = 0; step < 55; step++) {
    const p = (low + high) / 2;
    let logTerm = logCombination + correct * Math.log(p) + (count - correct) * Math.log1p(-p);
    let tail = Math.exp(logTerm);
    for (let k = correct; k < count; k++) {
      logTerm += Math.log(count - k) - Math.log(k + 1) + Math.log(p) - Math.log1p(-p);
      tail += Math.exp(logTerm);
    }
    if (tail < alpha) low = p; else high = p;
  }
  return low;
}

export function selectAcceptance(probabilities: number[][], labels: number[], known: boolean[], targetAccuracy: number,
  enabled = true) {
  if (probabilities.length !== labels.length || known.length !== labels.length) throw new Error('Acceptance inputs and labels must match.');
  // Fixed thresholds, with a family-wise correction for searching all of them.
  // Rows sharing a group are excluded by the caller: repeated correlated examples
  // cannot count as independent trials for a binomial confidence bound.
  const candidates = THRESHOLDS.map(threshold => {
    let accepted = 0, correct = 0;
    probabilities.forEach((p, i) => {
      if (known[i] && Math.max(...p) >= threshold) { accepted++; correct += Number(argmax(p) === labels[i]); }
    });
    return { threshold, accepted, correct, accuracy: accepted ? correct / accepted : null,
      coverage: labels.length ? accepted / labels.length : 0,
      lowerBound: accuracyLowerBound(correct, accepted, 0.05 / THRESHOLDS.length) };
  });
  const sufficient = enabled && labels.length >= MIN_CALIBRATION_EXAMPLES;
  const best = sufficient ? candidates.filter(c => c.accepted >= 30 && c.lowerBound >= targetAccuracy)
    .sort((a, b) => b.accepted - a.accepted || a.threshold - b.threshold)[0] : undefined;
  const status: ThresholdRecommendation['status'] = best ? 'ready' : sufficient ? 'target_not_met' : 'insufficient_data';
  const acceptance = { mode: 'automatic' as const, status, targetAccuracy };
  return { acceptance, threshold: best?.threshold ?? 1, examples: labels.length, confidenceLevel: 0.95,
    method: 'one-sided exact binomial; Bonferroni correction over a fixed threshold grid', candidates };
}

export function calibratedProbabilities(logits: number[][], temperature: number) {
  return logits.map(row => softmax(row, temperature));
}

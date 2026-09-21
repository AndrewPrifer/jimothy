import { argmax } from '../../dist/data.js';
import { fitTemperature, calibratedProbabilities, selectAcceptance, accuracyLowerBound, hardLogLoss } from '../../dist/calibration.js';
import { metrics } from '../../dist/metrics.js';

export function validateLogits(rows, examples, classes) {
  if (!Array.isArray(rows) || rows.length !== examples.length) throw new Error('Logits must cover every example exactly once.');
  return rows.map((row, i) => {
    if (row.id !== examples[i].id) throw new Error(`Unexpected example ID at row ${i}.`);
    if (!Array.isArray(row.logits) || row.logits.length !== classes || row.logits.some(x => typeof x !== 'number' || !Number.isFinite(x))) {
      throw new Error(`Invalid logits at row ${i}.`);
    }
    return row.logits;
  });
}

// No test data is accepted here. The head and threshold grid remain unchanged.
export function fitBrowserPolicy(calibrationRows, acceptanceRows, development, manifest) {
  const classes = manifest.task.labels.length;
  const calibrationLogits = validateLogits(calibrationRows, development.calibration, classes);
  const acceptanceLogits = validateLogits(acceptanceRows, development.acceptance, classes);
  const fit = fitTemperature(calibrationLogits, development.calibration.map(e => argmax(e.target)));
  const probabilities = calibratedProbabilities(acceptanceLogits, fit.temperature);
  const groups = new Set();
  const representatives = development.acceptance.flatMap((e, i) => {
    const group = e.group === undefined ? `text:${e.text}` : `group:${e.group}`;
    if (groups.has(group)) return [];
    groups.add(group); return [i];
  });
  const selection = selectAcceptance(representatives.map(i => probabilities[i]),
    representatives.map(i => argmax(development.acceptance[i].target)), representatives.map(() => true),
    manifest.thresholdRecommendation.targetAccuracy, fit.status === 'fitted');
  return {
    calibration: { method: fit.method, temperature: fit.temperature, status: fit.status },
    threshold: selection.threshold, acceptance: selection.acceptance,
    fit, selection,
  };
}

export function evaluatePolicy(logits, examples, labels, policy) {
  const probabilities = calibratedProbabilities(logits, policy.calibration.temperature);
  const enabled = policy.acceptance.status === 'ready';
  const result = metrics(examples, probabilities, labels, policy.threshold, undefined, enabled);
  const bins = Array.from({ length: 10 }, (_, i) => ({ from: i / 10, to: (i + 1) / 10, count: 0, confidenceSum: 0, correct: 0 }));
  let acceptedCorrect = 0;
  const decisions = probabilities.map((p, i) => {
    const winner = argmax(p), confidence = p[winner], correct = winner === argmax(examples[i].target);
    const accepted = enabled && confidence >= policy.threshold;
    acceptedCorrect += +(accepted && correct);
    const bin = bins[Math.min(9, Math.floor(confidence * 10))];
    bin.count++; bin.confidenceSum += confidence; bin.correct += +correct;
    return { choice: labels[winner], accepted, maxProbability: confidence };
  });
  const reliabilityBins = bins.map(bin => ({ from: bin.from, to: bin.to, count: bin.count,
    accuracy: bin.count ? bin.correct / bin.count : null, confidence: bin.count ? bin.confidenceSum / bin.count : null }));
  const accepted = result.operatingPoint.accepted;
  return { decisions, summary: {
    examples: examples.length, accuracy: result.targetAgreement,
    correct: decisions.filter((p, i) => p.choice === labels[argmax(examples[i].target)]).length,
    logLoss: hardLogLoss(logits, examples.map(e => argmax(e.target)), policy.calibration.temperature),
    brier: result.softTargetBrier,
    ece10: reliabilityBins.reduce((sum, bin) => sum + (bin.count ? bin.count / examples.length * Math.abs(bin.confidence - bin.accuracy) : 0), 0),
    accepted, acceptedCorrect, coverage: result.operatingPoint.coverage,
    acceptedAccuracy: result.operatingPoint.targetAgreement,
    acceptedAccuracyLower95: accuracyLowerBound(acceptedCorrect, accepted),
    reliabilityBins,
  } };
}

export function verifyBrowserDecisions(rows, decisions) {
  if (rows.length !== decisions.length) throw new Error('Decision counts differ.');
  rows.forEach((row, i) => {
    const expected = decisions[i];
    if (row.choice !== expected.choice || row.accepted !== expected.accepted ||
      typeof row.maxProbability !== 'number' || !Number.isFinite(row.maxProbability) ||
      Math.abs(row.maxProbability - expected.maxProbability) > 1e-10) {
      throw new Error(`Browser did not apply the saved policy at row ${i}.`);
    }
  });
}

import { argmax } from './data.js';
import { crossEntropy } from './linear.js';
import type { Example } from './types.js';

export function metrics(examples: Example[], predictions: number[][], labels: string[], threshold: number | null, known?: boolean[], acceptanceEnabled = true) {
  const count = examples.length;
  if (count === 0 || predictions.length !== count) throw new Error('Evaluation requires a non-empty matched prediction set.');
  const confusion = labels.map(() => labels.map(() => 0));
  let correct = 0, loss = 0, brier = 0, humanCount = 0, humanCorrect = 0, teacherCount = 0, teacherCorrect = 0;
  examples.forEach((example, i) => {
    const actual = argmax(example.target), predicted = argmax(predictions[i]);
    confusion[actual][predicted]++;
    correct += Number(actual === predicted);
    loss += crossEntropy(predictions[i], example.target);
    brier += example.target.reduce((sum, value, c) => sum + (value - predictions[i][c]) ** 2, 0);
    if (example.humanLabel !== undefined) { humanCount++; humanCorrect += Number(example.humanLabel === predicted); }
    if (example.teacher) { teacherCount++; teacherCorrect += Number(argmax(example.teacher) === predicted); }
  });
  const perClass = labels.map((label, c) => {
    const support = confusion[c].reduce((a, b) => a + b, 0);
    const predicted = confusion.reduce((sum, row) => sum + row[c], 0);
    const precision = predicted ? confusion[c][c] / predicted : 0;
    const recall = support ? confusion[c][c] / support : 0;
    return { label, support, precision, recall, f1: precision + recall ? 2 * precision * recall / (precision + recall) : 0 };
  });
  const atThreshold = (cutoff: number, enabled = true) => {
    let accepted = 0, matches = 0, goldAccepted = 0, goldCorrect = 0;
    predictions.forEach((p, i) => {
      if (enabled && (known?.[i] ?? true) && Math.max(...p) >= cutoff) {
        accepted++;
        matches += Number(argmax(p) === argmax(examples[i].target));
        if (examples[i].humanLabel !== undefined) { goldAccepted++; goldCorrect += Number(argmax(p) === examples[i].humanLabel); }
      }
    });
    return { threshold: cutoff, accepted, coverage: accepted / count,
      targetAgreement: accepted ? matches / accepted : null,
      humanExamplesAccepted: goldAccepted, humanAccuracy: goldAccepted ? goldCorrect / goldAccepted : null };
  };
  return {
    count, targetAgreement: correct / count,
    teacherAgreement: teacherCount ? teacherCorrect / teacherCount : null, teacherExamples: teacherCount,
    humanAccuracy: humanCount ? humanCorrect / humanCount : null, humanExamples: humanCount,
    softTargetCrossEntropy: loss / count, softTargetBrier: brier / count,
    macroF1: perClass.reduce((sum, row) => sum + row.f1, 0) / labels.length,
    perClass, confusionMatrix: { labels, rowsAre: 'reference', values: confusion },
    operatingPoint: threshold === null ? null : atThreshold(threshold, acceptanceEnabled),
    coverageCurve: [...new Set([0, 0.5, 0.7, 0.8, 0.9, 0.95, ...(threshold === null ? [] : [threshold])])].sort((a, b) => a - b).map(cutoff => atThreshold(cutoff)),
  };
}

/** Plain-language default report; detailed provenance and diagnostic metrics stay available. */
export function summarize(result: ReturnType<typeof metrics>) {
  return { reference: result.humanExamples === 0 && result.teacherExamples === result.count ? 'teacher labels' : 'provided labels', examples: result.count, accuracy: result.targetAgreement,
    coverage: result.operatingPoint?.coverage ?? null, acceptedAccuracy: result.operatingPoint?.targetAgreement ?? null };
}

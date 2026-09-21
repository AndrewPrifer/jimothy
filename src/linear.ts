import type { Head, Vector } from './types.js';

export function headLogits(head: Head, vector: Vector): number[] {
  return head.bias.map((bias, c) => {
    let score = bias;
    for (let j = 0; j < vector.indices.length; j++) score += head.weights[c][vector.indices[j]] * vector.values[j];
    return score;
  });
}
export function softmax(logits: number[], temperature = 1): number[] {
  const max = Math.max(...logits);
  const exp = logits.map(v => Math.exp((v - max) / temperature));
  const sum = exp.reduce((a, b) => a + b, 0);
  return exp.map(v => v / sum);
}
export function predictHead(head: Head, vector: Vector, temperature = 1): number[] {
  return softmax(headLogits(head, vector), temperature);
}
export function crossEntropy(probabilities: number[], target: number[]): number {
  return -target.reduce((sum, value, i) => sum + value * Math.log(Math.max(probabilities[i], 1e-12)), 0);
}
export function trainHead(
  vectors: Vector[], targets: number[][], dimensions: number,
  validation: { vectors: Vector[]; targets: number[][] },
  options: { epochs: number; learningRate: number; l2: number },
): { head: Head; epochs: number; bestEpoch: number; validationLoss: number } {
  const classes = targets[0].length;
  const head: Head = { weights: Array.from({ length: classes }, () => Array(dimensions).fill(0)), bias: Array(classes).fill(0) };
  const size = classes * (dimensions + 1);
  const m = new Float64Array(size), v = new Float64Array(size), gradient = new Float64Array(size);
  let best = structuredClone(head), bestLoss = Infinity, bestEpoch = 0, ran = 0;
  // Full-batch Adam is deterministic. Frozen features make small task heads inexpensive to fit.
  for (let epoch = 1; epoch <= options.epochs; epoch++) {
    gradient.fill(0);
    for (let i = 0; i < vectors.length; i++) {
      const probabilities = predictHead(head, vectors[i]);
      for (let c = 0; c < classes; c++) {
        const error = (probabilities[c] - targets[i][c]) / vectors.length;
        const offset = c * (dimensions + 1);
        gradient[offset + dimensions] += error;
        for (let j = 0; j < vectors[i].indices.length; j++) gradient[offset + vectors[i].indices[j]] += error * vectors[i].values[j];
      }
    }
    const correction1 = 1 - 0.9 ** epoch, correction2 = 1 - 0.999 ** epoch;
    for (let c = 0; c < classes; c++) {
      for (let d = 0; d <= dimensions; d++) {
        const index = c * (dimensions + 1) + d;
        const weight = d === dimensions ? head.bias[c] : head.weights[c][d];
        const g = gradient[index] + (d === dimensions ? 0 : options.l2 * weight);
        m[index] = 0.9 * m[index] + 0.1 * g;
        v[index] = 0.999 * v[index] + 0.001 * g * g;
        const next = weight - options.learningRate * (m[index] / correction1) / (Math.sqrt(v[index] / correction2) + 1e-8);
        if (d === dimensions) head.bias[c] = next; else head.weights[c][d] = next;
      }
    }
    const loss = validation.vectors.reduce((sum, vector, i) => sum + crossEntropy(predictHead(head, vector), validation.targets[i]), 0) / validation.vectors.length;
    if (!Number.isFinite(loss)) throw new Error('Training diverged; reduce --learning-rate.');
    ran = epoch;
    if (loss < bestLoss - 1e-7) { bestLoss = loss; best = structuredClone(head); bestEpoch = epoch; }
    if (epoch - bestEpoch >= 30) break;
  }
  return { head: best, epochs: ran, bestEpoch, validationLoss: bestLoss };
}

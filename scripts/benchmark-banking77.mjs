#!/usr/bin/env node
// Run from the repository root after building and preparing BANKING77 + MiniLM.
import { mkdir, readdir, readFile, stat, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { cpus, platform, arch, release } from 'node:os';
import { performance } from 'node:perf_hooks';
import { train, loadDataset } from '../dist/train.js';
import { loadClassifier } from '../dist/index.js';

const output = resolve(process.argv[2] ?? 'models/banking77-auto-v2');
const data = resolve('datasets/banking77');
const settings = { epochs: 100, learningRate: 0.05, maxFeatures: 4096, targetAccuracy: 0.95, seed: 42 };

async function directoryBytes(path) {
  let bytes = 0;
  for (const entry of await readdir(path, { withFileTypes: true })) {
    const child = join(path, entry.name);
    bytes += entry.isDirectory() ? await directoryBytes(child) : (await stat(child)).size;
  }
  return bytes;
}
function summary(metrics) {
  const { count, humanAccuracy, macroF1, softTargetCrossEntropy, operatingPoint, coverageCurve } = metrics;
  return { count, humanAccuracy, macroF1, softTargetCrossEntropy, operatingPoint, coverageCurve };
}

// Verify the separate Jev-compatible input/output path produces the same targets.
const task = join(data, 'task.json');
const human = await loadDataset({ task, data: join(data, 'train.jsonl') });
const converted = await loadDataset({ task, inputs: join(data, 'train.inputs.jsonl'), outputs: join(data, 'train.outputs.jsonl') });
if (human.examples.length !== converted.examples.length || human.examples.some((row, i) =>
  row.text !== converted.examples[i].text || JSON.stringify(row.target) !== JSON.stringify(converted.examples[i].target))) {
  throw new Error('Converted Jev-compatible records do not match the human-labelled dataset.');
}
const testRows = (await readFile(join(data, 'test.jsonl'), 'utf8')).trim().split('\n').map(JSON.parse);
await mkdir(dirname(output), { recursive: true });
await mkdir(output, { recursive: false }); // Never replace a previous benchmark.
const result = {
  createdAt: new Date().toISOString(),
  dataset: JSON.parse(await readFile(join(data, 'manifest.json'), 'utf8')),
  environment: { node: process.version, platform: platform(), arch: arch(), osRelease: release(), cpu: cpus()[0]?.model },
  settings,
  protocol: 'Automatic L2 tuning, temperature calibration, and acceptance selection on disjoint development subsets; cleaned official test evaluated after fitting. This is a follow-up on a previously inspected benchmark test set, not a new untouched holdout.',
  convertedPairsVerified: converted.examples.length,
  latencyProtocol: 'One loaded model at a time; 20 warm-up calls then 200 sequential single-input SDK predictions spaced across test inputs. Includes tokenization, features, and head; excludes model loading. Local CPU timing, not Jev comparison.',
  models: {},
};
for (const backend of ['tfidf', 'minilm']) {
  const destination = join(output, backend);
  const trained = await train({
    task, data: join(data, 'train.jsonl'), validation: join(data, 'validation.jsonl'), test: join(data, 'test.jsonl'),
    out: destination, backend, ...(backend === 'minilm' ? { encoder: resolve('.cache/minilm') } : {}), ...settings,
    onProgress: message => console.error(`[${new Date().toISOString()}] ${backend}: ${message}`),
  });
  const loadStart = performance.now();
  const classifier = await loadClassifier(destination);
  const loadMs = performance.now() - loadStart;
  const times = [];
  let sdkVerification;
  try {
    for (let i = 0; i < 20; i++) await classifier.predict(testRows[i].state);
    for (let i = 0; i < 200; i++) {
      const started = performance.now();
      await classifier.predict(testRows[Math.floor(i * testRows.length / 200)].state);
      times.push(performance.now() - started);
    }
    // Benchmark policy lives here, outside inference.
    const cutoff = classifier.metadata.thresholdRecommendation.threshold;
    const accepted = p => cutoff !== null && p.maxProbability >= cutoff;
    const predictions = await classifier.predictBatch(testRows.map(row => row.state));
    sdkVerification = { examples: predictions.length,
      correct: predictions.filter((p, i) => p.answer.choice === testRows[i].label).length,
      accepted: predictions.filter(accepted).length,
      acceptedCorrect: predictions.filter((p, i) => accepted(p) && p.answer.choice === testRows[i].label).length };
    if (sdkVerification.correct / testRows.length !== trained.report.test.humanAccuracy ||
      sdkVerification.accepted !== (trained.report.test.operatingPoint?.accepted ?? 0) ||
      (sdkVerification.accepted ? sdkVerification.acceptedCorrect / sdkVerification.accepted : null) !== (trained.report.test.operatingPoint?.humanAccuracy ?? null)) {
      throw new Error('Reloaded SDK predictions disagree with the training report.');
    }
  } finally { await classifier.dispose(); }
  times.sort((a, b) => a - b);
  result.models[backend] = {
    modelId: trained.manifest.id, training: trained.report.training,
    summary: trained.report.summary, development: trained.report.dataset.development,
    calibration: trained.report.calibration, thresholdRecommendation: trained.manifest.thresholdRecommendation,
    thresholdSelection: trained.report.thresholdSelection,
    sdkVerification,
    validation: summary(trained.report.validation), test: summary(trained.report.test),
    bundleBytes: await directoryBytes(destination), loadMs,
    singleInputLatencyMs: { count: times.length, p50: times[Math.ceil(times.length * 0.5) - 1], p95: times[Math.ceil(times.length * 0.95) - 1] },
  };
  await writeFile(join(output, 'results.json'), JSON.stringify(result, null, 2) + '\n');
  console.log(JSON.stringify({ backend, ...result.models[backend] }, null, 2));
}
console.error(`Saved benchmark results to ${join(output, 'results.json')}`);

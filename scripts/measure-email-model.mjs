// Re-measure an existing email model and its saved labels entirely offline.
import { readFile, writeFile, readdir, stat, mkdir } from 'node:fs/promises';
import { join, resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { cpus, platform, arch } from 'node:os';
import { AutoTokenizer } from '@huggingface/transformers';
import { readRows } from '../dist/data.js';
import { loadDataset } from '../dist/train.js';
import { metrics, summarize } from '../dist/metrics.js';
import { loadClassifier } from '../dist/index.js';

export async function measureEmailModel({ dataset, out, reportFile, run = null }) {
  dataset = resolve(dataset); out = resolve(out);
  const report = JSON.parse(await readFile(join(out, 'report.json'), 'utf8'));
  const test = await loadDataset({ data: join(dataset, 'test.jsonl') }, report.task);
  const all = await loadDataset({ inputs: join(dataset, 'inputs.jsonl'), outputs: join(dataset, 'outputs.jsonl') }, report.task);
  const outputs = await readRows(join(dataset, 'outputs.jsonl'));
  if (all.examples.length !== 300 || test.examples.length !== 60) throw new Error('Expected 300 labeled emails and 60 test examples.');
  const tokenizer = await AutoTokenizer.from_pretrained(join(out, 'encoder'), { local_files_only: true });
  const tokens = all.examples.map(row => tokenizer(row.text, { truncation: false, padding: false, return_tensor: false }).input_ids.length);
  const started = performance.now();
  const classifier = await loadClassifier(out);
  const loadMs = performance.now() - started;
  const latencies = [], predictions = [];
  let batch, batchMs;
  try {
    for (const row of test.examples.slice(0, 10)) await classifier.predict(row.state);
    for (const row of test.examples) {
      const start = performance.now();
      predictions.push(await classifier.predict(row.state));
      latencies.push(performance.now() - start);
    }
    const start = performance.now();
    batch = await classifier.predictBatch(test.examples.map(row => row.state));
    batchMs = performance.now() - start;
  } finally { await classifier.dispose(); }
  const labels = report.task.labels;
  const score = rows => metrics(test.examples, rows.map(row => labels.map(label => row.answer.probabilities[label])),
    labels, classifier.metadata.thresholdRecommendation.threshold);
  const singleMetrics = score(predictions), batchMetrics = score(batch);
  if (JSON.stringify(batchMetrics.confusionMatrix) !== JSON.stringify(report.test.confusionMatrix)) throw new Error('Reloaded batched SDK disagrees with the exported training evaluation.');
  // Dynamic q8 quantization can depend on other inputs in the encoder batch. Report this explicitly.
  const changedIds = test.examples.flatMap((row, i) => predictions[i].answer.choice === batch[i].answer.choice ? [] : [row.id]);
  const maxProbabilityDifference = Math.max(...predictions.flatMap((row, i) => labels.map(label => Math.abs(row.answer.probabilities[label] - batch[i].answer.probabilities[label]))));
  await writeFile(join(dataset, 'test.predictions.jsonl'), test.examples.map((row, i) => JSON.stringify({
    id: row.id, ...predictions[i], batchAnswer: batch[i].answer,
  })).join('\n') + '\n');
  const winner = probabilities => labels.reduce((best, label) => probabilities[label] > probabilities[best] ? label : best);
  const counts = Object.fromEntries(labels.map(label => [label, 0]));
  const targets = new Map((await readRows(join(dataset, 'generation-targets.jsonl'))).map(row => [row.id, row.intendedCategory]));
  let generatorTargetMatches = 0, inputTokens = 0, outputTokens = 0, gatewayCost = 0, marketCost = 0, costRecords = 0, marketCostRecords = 0;
  for (const row of outputs) {
    const label = winner(row.response.answers.category.probabilities);
    counts[label]++;
    generatorTargetMatches += Number(label === targets.get(row.id));
    inputTokens += row.response.usage?.input_tokens ?? 0;
    outputTokens += row.response.usage?.output_tokens ?? 0;
    const cost = row.response.provider_metadata?.gateway?.cost;
    if (cost !== undefined && Number.isFinite(Number(cost))) { costRecords++; gatewayCost += Number(cost); }
    const market = row.response.provider_metadata?.gateway?.marketCost;
    if (market !== undefined && Number.isFinite(Number(market))) { marketCostRecords++; marketCost += Number(market); }
  }
  async function bytes(path) {
    const entry = await stat(path);
    if (!entry.isDirectory()) return entry.size;
    return (await Promise.all((await readdir(path)).map(name => bytes(join(path, name))))).reduce((a, b) => a + b, 0);
  }
  latencies.sort((a, b) => a - b);
  const result = {
    hardware: { cpu: cpus()[0].model, platform: platform(), arch: arch(), node: process.version },
    generator: JSON.parse(await readFile(join(dataset, 'generation-report.json'), 'utf8')),
    teacher: { model: report.teacher.model, resolvedModels: [...new Set(outputs.map(row => row.response.model))],
      count: 300, distribution: counts, generatorTargetMatches,
      note: 'Generator targets describe intended categories, not human reference labels. Training-run counters exclude earlier attempts and test labeling.',
      trainingRun: report.teacher, usage: { inputTokens, outputTokens },
      gatewayCostUsd: costRecords === 300 ? gatewayCost : null, marketCostUsd: marketCostRecords === 300 ? marketCost : null },
    local: { model: classifier.metadata.id, trainingMs: report.training.elapsedMs, bundleBytes: await bytes(out), loadMs,
      wordpieces: { min: Math.min(...tokens), max: Math.max(...tokens) },
      inference: { samples: 60, warmups: 10, p50Ms: latencies[29], p95Ms: latencies[56], batchMs },
      dataset: report.dataset, summary: report.summary, thresholdRecommendation: classifier.metadata.thresholdRecommendation,
      single: { ...summarize(singleMetrics), macroF1: singleMetrics.macroF1, perClass: singleMetrics.perClass },
      batch: { ...summarize(batchMetrics), macroF1: batchMetrics.macroF1, perClass: batchMetrics.perClass },
      batchVsSingle: { changedLabels: changedIds.length, changedIds, maxProbabilityDifference },
      calibration: report.calibration, warnings: [...new Set([...report.warnings,
        ...(batchMetrics.perClass.some(row => row.support === 0) ? ['Test data does not cover every label; performance for absent labels is not established.'] : [])])] },
    run,
    timingNote: 'CPU q8 inference; model load in a warm process; 60 sequential single-input predictions after 10 warmups. Training-run labeling time covers that resumed run only, excluding earlier attempts and test labeling. A null run means the model was measured after training in a separate invocation.',
  };
  const path = resolve(reportFile);
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, JSON.stringify(result, null, 2) + '\n');
  return result;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const { values } = parseArgs({ options: {
    dataset: { type: 'string', default: 'datasets/email-300' }, model: { type: 'string', default: 'models/email-300/minilm' },
    report: { type: 'string', default: 'benchmarks/email-300.json' },
  } });
  const result = await measureEmailModel({ dataset: values.dataset, out: values.model, reportFile: values.report });
  console.log(JSON.stringify({ report: values.report, summary: result.local.summary, single: result.local.single.accuracy,
    inference: result.local.inference, batchVsSingle: result.local.batchVsSingle }, null, 2));
}

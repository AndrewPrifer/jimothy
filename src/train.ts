import { mkdir, mkdtemp, readFile, rename, rm, stat, writeFile } from 'node:fs/promises';
import { basename, dirname, isAbsolute, join, relative, resolve } from 'node:path';
import { performance } from 'node:perf_hooks';
import { writeManifest, readManifest } from './artifact.js';
import { argmax, assertDisjoint, datasetDigest, joinFiles, parseDataset, readRows, splitDevelopment, splitExamples } from './data.js';
import { calibratedProbabilities, fitTemperature, selectAcceptance } from './calibration.js';
import { headLogits, predictHead, trainHead } from './linear.js';
import { metrics, summarize } from './metrics.js';
import { minilmExtractor, prepareEncoder } from './minilm.js';
import { parseTask, probability } from './schema.js';
import { fitTfidf, tfidfExtractor } from './tfidf.js';
import { labelInputs, readTeacherInputs, teacherEndpoint } from './teacher.js';
import type { Example, FeatureExtractor, Task, ThresholdRecommendation } from './types.js';

export interface DatasetOptions { data?: string; inputs?: string; outputs?: string; task?: string; question?: string }
export interface TrainOptions extends DatasetOptions {
  out: string;
  backend?: 'minilm' | 'tfidf';
  encoder?: string;
  validation?: string;
  test?: string;
  epochs?: number;
  learningRate?: number;
  l2?: number;
  maxFeatures?: number;
  targetAccuracy?: number;
  seed?: number;
  teacher?: string;
  teacherUrl?: string;
  teacherKeyEnv?: string;
  teacherCache?: string;
  teacherRpm?: number;
  onProgress?: (message: string) => void;
}
export async function loadDataset(options: DatasetOptions, task?: Task) {
  if ((options.data ? 1 : 0) + (options.inputs || options.outputs ? 1 : 0) !== 1 || (!!options.inputs !== !!options.outputs)) {
    throw new Error('Provide either --data, or both --inputs and --outputs.');
  }
  const explicit = options.task ? parseTask(JSON.parse(await readFile(options.task, 'utf8')), options.question) : task;
  const rows = options.data ? await readRows(options.data) : await joinFiles(options.inputs!, options.outputs!);
  return parseDataset(rows, explicit, options.question);
}
async function unusedOutput(path: string): Promise<void> {
  try { await stat(path); }
  catch (error) { if ((error as NodeJS.ErrnoException).code === 'ENOENT') return; throw error; }
  throw new Error(`Output already exists: ${path}. Choose a new directory to preserve the existing model.`);
}
function integer(value: number, min: number, max: number, name: string): number {
  if (!Number.isInteger(value) || value < min || value > max) throw new Error(`${name} must be an integer between ${min} and ${max}.`);
  return value;
}
export async function train(options: TrainOptions) {
  if (!options.out) throw new Error('--out is required.');
  const backend = options.backend ?? 'minilm';
  if (!['minilm', 'tfidf'].includes(backend)) throw new Error('--backend must be minilm or tfidf.');
  if (options.encoder && backend !== 'minilm') throw new Error('--encoder only applies to the minilm backend.');
  const epochs = integer(options.epochs ?? 200, 1, 10_000, 'epochs');
  const maxFeatures = integer(options.maxFeatures ?? 4096, 1, 100_000, 'max-features');
  const seed = integer(options.seed ?? 42, 0, 2 ** 32 - 1, 'seed');
  const learningRate = options.learningRate ?? 0.05;
  if (!Number.isFinite(learningRate) || learningRate <= 0 || learningRate > 1) throw new Error('learning-rate must be greater than 0 and at most 1.');
  if (options.l2 !== undefined && (!Number.isFinite(options.l2) || options.l2 < 0 || options.l2 > 1)) throw new Error('l2 must be between 0 and 1.');
  const regularizations = options.l2 === undefined ? [0.001, 0.0001, 0.00001, 0.000001, 0] : [options.l2];
  const targetAccuracy = probability(options.targetAccuracy ?? 0.95, 'target-accuracy');
  if (targetAccuracy === 0) throw new Error('target-accuracy must be greater than 0.');
  const destination = resolve(options.out);
  await unusedOutput(destination);
  options.onProgress?.('Validating the task and dataset…');
  if (options.teacher === undefined && [options.teacherUrl, options.teacherKeyEnv, options.teacherCache, options.teacherRpm].some(v => v !== undefined)) {
    throw new Error('--teacher-url, --teacher-key-env, --teacher-cache and --teacher-rpm require --teacher.');
  }
  let dataset: Awaited<ReturnType<typeof loadDataset>>;
  let teacher: Awaited<ReturnType<typeof labelInputs>>['summary'] | undefined;
  if (options.teacher !== undefined) {
    if (!options.inputs || options.data !== undefined || options.outputs !== undefined) {
      throw new Error('Use --teacher with --inputs, without --data or --outputs.');
    }
    const teacherCache = resolve(options.teacherCache ?? `${destination}.teacher`);
    const contains = (parent: string, child: string) => {
      const path = relative(parent, child);
      return path === '' || (!isAbsolute(path) && path !== '..' && !path.startsWith('../'));
    };
    if (contains(destination, teacherCache) || contains(teacherCache, destination)) throw new Error('Keep --teacher-cache outside the model output directory, and vice versa.');
    const teacherOptions = { teacher: options.teacher, teacherUrl: options.teacherUrl, teacherKeyEnv: options.teacherKeyEnv, teacherCache, teacherRpm: options.teacherRpm, onProgress: options.onProgress };
    teacherEndpoint(teacherOptions);
    const explicit = options.task ? parseTask(JSON.parse(await readFile(options.task, 'utf8')), options.question) : undefined;
    const inputs = await readTeacherInputs(options.inputs, explicit, options.question);
    if (inputs.uniqueCount < 10) throw new Error('Provide at least 10 unique examples for a train/validation experiment. This minimum is not a quality guarantee.');
    // Detect malformed held-out data and leakage before spending on labels.
    const validation = options.validation ? await loadDataset({ data: options.validation }, inputs.task) : undefined;
    const test = options.test ? await loadDataset({ data: options.test }, inputs.task) : undefined;
    if (validation) assertDisjoint(inputs.inputs, validation.examples, 'Validation data');
    if (test) {
      assertDisjoint(inputs.inputs, test.examples, 'Test data');
      if (validation) assertDisjoint(validation.examples, test.examples, 'Test data');
    }
    const labeled = await labelInputs(inputs.inputs, inputs.task, teacherOptions);
    teacher = labeled.summary;
    // Train from this run's responses; another run may now reuse the cache and replace its outputs file.
    dataset = parseDataset(inputs.inputs.map((input, i) => ({ id: input.id, state: input.state,
      ...(input.group === undefined ? {} : { group: input.group }), response: labeled.rows[i].response })), inputs.task);
  } else dataset = await loadDataset(options);
  if (dataset.examples.length < 10) throw new Error('Provide at least 10 unique examples for a train/validation experiment. This minimum is not a quality guarantee.');
  const split = options.validation
    ? { train: dataset.examples, validation: (await loadDataset({ data: options.validation }, dataset.task)).examples }
    : splitExamples(dataset.examples, seed);
  assertDisjoint(split.train, split.validation, 'Validation data');
  const development = splitDevelopment(split.validation, seed);
  const test = options.test ? (await loadDataset({ data: options.test }, dataset.task)).examples : undefined;
  if (test) { assertDisjoint(split.train, test, 'Test data'); assertDisjoint(split.validation, test, 'Test data'); }
  const classes = new Set(split.train.map(e => argmax(e.target)));
  if (classes.size < 2) throw new Error('Training data must cover at least two winning labels. Add negative or alternative-class examples.');
  const warnings: string[] = [];
  if (classes.size < dataset.task.labels.length) warnings.push('Some labels have no winning training examples; their performance is not established.');
  if (new Set(split.validation.map(e => argmax(e.target))).size < dataset.task.labels.length) warnings.push('Validation does not cover every label.');
  if (!test) warnings.push('No separate test set supplied. Development results were used for selection; evaluate on new examples before deployment.');
  if (test && new Set(test.map(e => argmax(e.target))).size < dataset.task.labels.length) warnings.push('Test data does not cover every label; performance for absent labels is not established.');
  const teacherModels = [...new Set(dataset.examples.flatMap(e => e.teacherModel ? [e.teacherModel] : []))].sort();
  if (teacherModels.length > 1) warnings.push('Dataset contains multiple teacher model versions; review whether they follow the same policy.');
  await mkdir(dirname(destination), { recursive: true });
  const staging = await mkdtemp(join(dirname(destination), `.${basename(destination)}-`));
  let extractor: FeatureExtractor | undefined;
  try {
    const started = performance.now();
    if (backend === 'minilm') {
      options.onProgress?.(options.encoder ? 'Bundling the local MiniLM encoder…' : 'Downloading public MiniLM encoder assets; training inputs stay local…');
      const config = await prepareEncoder(join(staging, 'encoder'), options.encoder);
      extractor = await minilmExtractor(config, staging);
    } else extractor = tfidfExtractor(fitTfidf(split.train.map(e => e.text), maxFeatures));
    options.onProgress?.(`Encoding ${split.train.length} training and ${split.validation.length} development examples once…`);
    const trainVectors = await extractor.encode(split.train.map(e => e.text));
    const validationVectors = await extractor.encode(development.tuning.map(e => e.text));
    const calibrationVectors = await extractor.encode(development.calibration.map(e => e.text));
    const acceptanceVectors = await extractor.encode(development.acceptance.map(e => e.text));
    const dimensions = extractor.config.kind === 'tfidf' ? extractor.config.vocabulary.length : extractor.config.dimensions;
    const candidates: { l2: number; epochs: number; bestEpoch: number; validationLoss: number }[] = [];
    let selected: { l2: number; fit: ReturnType<typeof trainHead> } | undefined;
    const targets = split.train.map(e => e.target);
    const validationTargets = development.tuning.map(e => e.target);
    for (const l2 of regularizations) {
      options.onProgress?.(`Fitting ${dataset.task.labels.length} classes on ${dimensions} features (L2 ${l2})…`);
      const fit = trainHead(trainVectors, targets, dimensions,
        { vectors: validationVectors, targets: validationTargets }, { epochs, learningRate, l2 });
      candidates.push({ l2, epochs: fit.epochs, bestEpoch: fit.bestEpoch, validationLoss: fit.validationLoss });
      if (!selected || fit.validationLoss < selected.fit.validationLoss) selected = { l2, fit };
    }
    const { fit, l2 } = selected!;
    options.onProgress?.('Calibrating probabilities and finding a recommended threshold…');
    const calibrationFit = fitTemperature(calibrationVectors.map(v => headLogits(fit.head, v)), development.calibration.map(e => argmax(e.target)));
    const calibration = { method: calibrationFit.method, temperature: calibrationFit.temperature, status: calibrationFit.status };
    const acceptanceProbabilities = calibratedProbabilities(acceptanceVectors.map(v => headLogits(fit.head, v)), calibration.temperature);
    // One deterministic representative per group prevents repeated customers/documents
    // from supplying spurious independent evidence for the binomial bound.
    const seenGroups = new Set<string>();
    const representatives = development.acceptance.flatMap((e, i) => {
      const key = e.group === undefined ? `text:${e.text}` : `group:${e.group}`;
      if (seenGroups.has(key)) return [];
      seenGroups.add(key);
      return [i];
    });
    const selection = selectAcceptance(representatives.map(i => acceptanceProbabilities[i]),
      representatives.map(i => argmax(development.acceptance[i].target)), representatives.map(() => true),
      targetAccuracy, calibration.status === 'fitted');
    const recommendation: ThresholdRecommendation = { threshold: selection.acceptance.status === 'ready' ? selection.threshold : null,
      status: selection.acceptance.status, targetAccuracy };
    if (recommendation.status === 'insufficient_data') warnings.push('Not enough independent held-out examples to recommend a threshold. Predictions remain available.');
    if (recommendation.status === 'target_not_met') warnings.push('No threshold met the requested accuracy target with sufficient evidence. Predictions remain available.');
    const validationMetrics = metrics(development.tuning, validationVectors.map(v => predictHead(fit.head, v, calibration.temperature)), dataset.task.labels, recommendation.threshold);
    const acceptanceMetrics = development.acceptance.length ? metrics(development.acceptance, acceptanceProbabilities, dataset.task.labels, recommendation.threshold) : null;
    options.onProgress?.('Evaluating the calibrated model…');
    const testVectors = test ? await extractor.encode(test.map(e => e.text)) : undefined;
    const testMetrics = test && testVectors ? metrics(test, testVectors.map(v => predictHead(fit.head, v, calibration.temperature)), dataset.task.labels, recommendation.threshold) : null;
    const report = {
      format: 'jev-distill-report', version: 3,
      task: dataset.task, backend, seed,
      ...(teacher ? { teacher } : {}),
      summary: { ...summarize(testMetrics ?? acceptanceMetrics ?? validationMetrics), evaluation: testMetrics ? 'test' : 'development' },
      thresholdRecommendation: recommendation,
      dataset: { uniqueExamples: dataset.examples.length, duplicatesRemoved: dataset.duplicatesRemoved, teacherModels,
        train: { count: split.train.length, sha256: datasetDigest(split.train) },
        validation: { count: split.validation.length, sha256: datasetDigest(split.validation), source: options.validation ? 'explicit' : 'grouped-80/20' },
        development: Object.fromEntries(Object.entries(development).map(([key, rows]) => [key, { count: rows.length, sha256: datasetDigest(rows) }])),
        test: test ? { count: test.length, sha256: datasetDigest(test) } : null },
      training: { epochs: fit.epochs, bestEpoch: fit.bestEpoch, learningRate, l2, dimensions, candidates, elapsedMs: performance.now() - started },
      calibration: { ...calibrationFit, reference: 'winning provided label', note: 'Fitting loss, not independent calibration performance.' },
      thresholdSelection: { ...recommendation, examples: selection.examples, confidenceLevel: selection.confidenceLevel,
        method: selection.method, candidates: selection.candidates,
        note: 'Evidence uses one representative per group; statistical bounds assume independent, representative groups and do not cover distribution shift.' },
      validation: validationMetrics,
      thresholdEvaluation: acceptanceMetrics,
      test: testMetrics,
      warnings,
    };
    await writeFile(join(staging, 'report.json'), JSON.stringify(report, null, 2) + '\n');
    const manifest = await writeManifest(staging, {
      format: 'jev-distill', version: 3, createdAt: new Date().toISOString(), task: dataset.task,
      preprocessing: 'canonical-json-v1', features: extractor.config, head: fit.head, calibration, thresholdRecommendation: recommendation,
    });
    await extractor.dispose();
    extractor = undefined;
    await readManifest(staging);
    await unusedOutput(destination);
    await rename(staging, destination);
    options.onProgress?.(`Exported ${manifest.id} to ${destination}`);
    return { directory: destination, manifest, report };
  } catch (error) {
    if (extractor) await extractor.dispose().catch(() => {});
    await rm(staging, { recursive: true, force: true });
    throw error;
  }
}
export async function evaluateBundle(directory: string, data: string) {
  const manifest = await readManifest(resolve(directory));
  const recommendation = manifest.thresholdRecommendation;
  const dataset = await loadDataset({ data }, manifest.task);
  const extractor = manifest.features.kind === 'tfidf' ? tfidfExtractor(manifest.features) : await minilmExtractor(manifest.features, resolve(directory));
  try {
    const started = performance.now();
    const vectors = await extractor.encode(dataset.examples.map((e: Example) => e.text));
    const predictions = vectors.map(v => predictHead(manifest.head, v, manifest.calibration.temperature));
    const result = metrics(dataset.examples, predictions, manifest.task.labels, recommendation.threshold);
    return { model: manifest.id, summary: summarize(result), thresholdRecommendation: recommendation, metrics: result,
      batchInferenceMs: performance.now() - started,
      note: 'Model load time is excluded. Evaluation data independence is the caller’s responsibility.' };
  } finally { await extractor.dispose(); }
}

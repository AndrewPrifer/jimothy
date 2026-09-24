import { canonical, object, parseTask, probability } from './schema-core.js';
import type { Manifest } from './types.js';
export const ENCODER_FILES = ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'onnx/model_quantized.onnx', 'README.md', 'LICENSE.txt', 'encoder-source.json'];

export function modelIdentity(manifest: Omit<Manifest, 'id'> | Manifest): string {
  return canonical({ version: manifest.version, task: manifest.task, features: manifest.features, head: manifest.head,
    calibration: manifest.calibration, thresholdRecommendation: manifest.thresholdRecommendation, files: manifest.files,
  });
}
export function parseManifest(value: unknown): Manifest {
  const raw = object(value, 'Model manifest');
  if (raw.format !== 'jev-distill' || raw.version !== 3 || raw.preprocessing !== 'canonical-json-v1') throw new Error('Unsupported model format/version/preprocessing. Train a new bundle with the current CLI.');
  const calibration = object(raw.calibration, 'calibration');
  if (calibration.method !== 'temperature' || !['fitted', 'insufficient_data'].includes(String(calibration.status)) ||
    typeof calibration.temperature !== 'number' || !Number.isFinite(calibration.temperature) || calibration.temperature < 0.05 || calibration.temperature > 20 ||
    (calibration.status === 'insufficient_data' && calibration.temperature !== 1)) throw new Error('Invalid probability calibration.');
  if (raw.threshold !== undefined || raw.acceptance !== undefined) throw new Error('Model bundles store threshold recommendations, not acceptance policies.');
  const recommendation = object(raw.thresholdRecommendation, 'thresholdRecommendation');
  if (!['ready', 'insufficient_data', 'target_not_met'].includes(String(recommendation.status)) ||
    typeof recommendation.targetAccuracy !== 'number' || !(recommendation.targetAccuracy > 0 && recommendation.targetAccuracy <= 1) ||
    (recommendation.status !== 'ready' && recommendation.threshold !== null)) throw new Error('Invalid threshold recommendation.');
  if (recommendation.status === 'ready') probability(recommendation.threshold, 'recommended threshold');
  const taskRaw = object(raw.task, 'task');
  if (typeof taskRaw.questionId !== 'string') throw new Error('Invalid task questionId.');
  const task = parseTask({ questions: { [taskRaw.questionId]: taskRaw.question } }, taskRaw.questionId);
  if (canonical(taskRaw.labels) !== canonical(task.labels)) throw new Error('Model label order does not match its task.');
  const features = object(raw.features, 'features');
  let dimensions: number;
  if (features.kind === 'tfidf') {
    if (!Array.isArray(features.vocabulary) || features.vocabulary.length < 1 || features.vocabulary.length > 100_000 ||
      features.vocabulary.some(v => typeof v !== 'string') || new Set(features.vocabulary).size !== features.vocabulary.length ||
      !Array.isArray(features.idf) || features.idf.length !== features.vocabulary.length || features.idf.some(v => typeof v !== 'number' || !Number.isFinite(v) || v <= 0) ||
      !Number.isInteger(features.maxTokens) || Number(features.maxTokens) < 1 || Number(features.maxTokens) > 100_000) throw new Error('Invalid TF-IDF feature configuration.');
    dimensions = features.vocabulary.length;
  } else if (features.kind === 'minilm') {
    if (features.directory !== 'encoder' || features.dimensions !== 384 || features.maxTokens !== 256 ||
        (features.longInput !== undefined && features.longInput !== 'chunk') || features.dtype !== 'q8' ||
        features.modelId !== 'Xenova/all-MiniLM-L6-v2' || typeof features.revision !== 'string' || !/^[a-f0-9]{40}$/.test(features.revision)) throw new Error('Invalid MiniLM feature configuration.');
    dimensions = 384;
  } else throw new Error('Unknown feature backend.');
  const head = object(raw.head, 'head');
  const finite = (v: unknown) => typeof v === 'number' && Number.isFinite(v);
  if (!Array.isArray(head.bias) || head.bias.length !== task.labels.length || !head.bias.every(finite) ||
      !Array.isArray(head.weights) || head.weights.length !== task.labels.length ||
      head.weights.some(row => !Array.isArray(row) || row.length !== dimensions || !row.every(finite))) throw new Error('Invalid classifier weight dimensions or values.');
  const files = object(raw.files, 'files');
  const expected = ['report.json', ...(features.kind === 'minilm' ? ENCODER_FILES.map(f => `encoder/${f}`) : [])].sort();
  if (canonical(Object.keys(files).sort()) !== canonical(expected)) throw new Error('Model bundle has an invalid file list.');
  for (const file of expected) {
    if (typeof files[file] !== 'string' || !/^[a-f0-9]{64}$/.test(files[file] as string)) throw new Error(`Invalid checksum for ${file}.`);
  }
  return raw as unknown as Manifest;
}

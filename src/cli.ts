#!/usr/bin/env node
import { parseArgs } from 'node:util';
import { mkdir, mkdtemp, readFile, rename, rm, stat } from 'node:fs/promises';
import { basename, dirname, join, resolve } from 'node:path';
import { readManifest } from './artifact.js';
import { readRows } from './data.js';
import { prepareBrowser } from './browser-assets.js';
import { prepareEncoder } from './minilm.js';
import { object, stateText } from './schema.js';
import { loadClassifier } from './sdk.js';
import { evaluateBundle, loadDataset, train } from './train.js';

const help = `jimothy — train local classifiers from Jev-compatible datasets

Usage:
  jimothy validate --task task.json --data examples.jsonl
  jimothy train --task task.json --data examples.jsonl --out model-dir
  jimothy train --task task.json --inputs inputs.jsonl --outputs outputs.jsonl --out model-dir
  jimothy train --task task.json --inputs inputs.jsonl --teacher typesafe-ai/jev --out model-dir
  jimothy predict --model model-dir --text "A new input"
  jimothy predict --model model-dir --state '{"message":"A new input"}'
  jimothy predict --model model-dir --data inputs.jsonl
  jimothy evaluate --model model-dir --data held-out.jsonl
  jimothy inspect --model model-dir
  jimothy prepare-encoder --out encoder-dir
  jimothy prepare-browser --out public/jimothy

Training options:
  --task FILE             Jev-compatible {questions:{...}} task definition
  --question ID           Select one question when requests contain several
  --data FILE             Combined JSONL or JSON array of input/output records
  --inputs/--outputs FILE Separate files, joined by required string ids
  --teacher MODEL        Label --inputs with a Jev-compatible teacher before training
  --teacher-url URL      TypeSafe-compatible endpoint (default: Vercel AI Gateway)
  --teacher-key-env NAME API key variable (default: AI_GATEWAY_API_KEY)
  --teacher-cache DIR    Resume saved labels (default: <out>.teacher)
  --teacher-rpm NUMBER   Optional maximum request starts per minute
  --backend minilm|tfidf  Frozen MiniLM (default) or word/bigram TF-IDF
  --encoder DIR          Reuse local encoder assets; offline unless --teacher is used
  --long-input chunk     Mean-pool token windows when MiniLM input exceeds 256 wordpieces
  --validation FILE      Development data; otherwise grouped 80/20 holdout
  --test FILE            Untouched test set, evaluated after model selection
  --target-accuracy N    Accuracy target for the recommended cutoff (default 0.95)
  --epochs NUMBER        Maximum full-batch Adam epochs (default 200)
  --learning-rate NUMBER Adam learning rate (default 0.05)
  --l2 NUMBER            Advanced: fix regularisation instead of automatic tuning
  --max-features NUMBER  TF-IDF vocabulary limit (default 4096)
  --seed NUMBER          Reproducible grouped split seed (default 42)

JSON results go to stdout; progress goes to stderr. Existing model directories are never overwritten.
The default backend downloads pretrained encoder assets if --encoder is omitted.
The SDK and predict/evaluate commands never call a model provider or download assets.
`;
const teacherFlags = ['teacher', 'teacher-url', 'teacher-key-env', 'teacher-cache', 'teacher-rpm'];
const names = ['task', 'question', 'data', 'inputs', 'outputs', 'out', 'backend', 'encoder', 'long-input', 'validation', 'test', 'target-accuracy', 'epochs', 'learning-rate', 'l2', 'max-features', 'seed', 'model', 'text', 'state', ...teacherFlags] as const;
const dataFlags = ['task', 'question', 'data', 'inputs', 'outputs'];
const allowed: Record<string, string[]> = {
  train: [...dataFlags, ...teacherFlags, 'out', 'backend', 'encoder', 'long-input', 'validation', 'test', 'target-accuracy', 'epochs', 'learning-rate', 'l2', 'max-features', 'seed'],
  validate: dataFlags,
  predict: ['model', 'text', 'state', 'data'],
  evaluate: ['model', 'data'],
  inspect: ['model'],
  'prepare-encoder': ['out'],
  'prepare-browser': ['out'],
};

async function main(): Promise<void> {
  const argv = process.argv.slice(2);
  if (!argv.length || argv[0] === '--help' || argv[0] === 'help') { console.log(help); return; }
  if (argv[0] === '--version') { console.log('0.1.0'); return; }
  const command = argv.shift()!;
  if (!allowed[command]) throw new Error(`Unknown command: ${command}. Use --help.`);
  const options = Object.fromEntries(names.map(name => [name, { type: 'string' as const }]));
  const { values } = parseArgs({ args: argv, options: { ...options, help: { type: 'boolean' } }, strict: true, allowPositionals: false });
  if (values.help) { console.log(help); return; }
  for (const key of Object.keys(values)) if (!allowed[command].includes(key)) throw new Error(`--${key} does not apply to ${command}.`);
  const string = (key: string): string | undefined => (values as Record<string, string | boolean | undefined>)[key] as string | undefined;
  const requireString = (key: string): string => { const value = string(key); if (!value) throw new Error(`--${key} is required.`); return value; };
  const number = (key: string): number | undefined => {
    const value = string(key);
    if (value === undefined) return undefined;
    if (!value.trim() || !Number.isFinite(Number(value))) throw new Error(`--${key} must be a finite number.`);
    return Number(value);
  };
  const print = (result: unknown) => console.log(JSON.stringify(result, null, 2));
  const datasetOptions = { data: string('data'), inputs: string('inputs'), outputs: string('outputs'), task: string('task'), question: string('question') };
  if (command === 'validate') {
    const dataset = await loadDataset(datasetOptions);
    print({ task: dataset.task, uniqueExamples: dataset.examples.length, duplicatesRemoved: dataset.duplicatesRemoved,
      humanLabels: dataset.examples.filter(e => e.humanLabel !== undefined).length,
      teacherResponses: dataset.examples.filter(e => e.teacher !== undefined).length });
  } else if (command === 'train') {
    const result = await train({ ...datasetOptions, out: requireString('out'), backend: string('backend') as 'minilm' | 'tfidf' | undefined,
      encoder: string('encoder'), longInput: string('long-input') as 'chunk' | undefined,
      validation: string('validation'), test: string('test'), targetAccuracy: number('target-accuracy'),
      epochs: number('epochs'), learningRate: number('learning-rate'), l2: number('l2'), maxFeatures: number('max-features'), seed: number('seed'),
      teacher: string('teacher'), teacherUrl: string('teacher-url'), teacherKeyEnv: string('teacher-key-env'), teacherCache: string('teacher-cache'),
      teacherRpm: number('teacher-rpm'),
      onProgress: message => console.error(message),
    });
    print({ directory: result.directory, model: result.manifest.id, backend: result.manifest.features.kind, report: join(result.directory, 'report.json'),
      summary: result.report.summary, thresholdRecommendation: result.report.thresholdRecommendation, warnings: result.report.warnings });
  } else if (command === 'predict') {
    const sources = ['text', 'state', 'data'].filter(key => string(key) !== undefined);
    if (sources.length !== 1) throw new Error('Provide exactly one of --text, --state (JSON), or --data.');
    const classifier = await loadClassifier(requireString('model'));
    try {
      if (string('data')) {
        for (const raw of await readRows(requireString('data'))) {
          const row = object(raw, 'Prediction row');
          const state = stateText(row.state).state;
          console.log(JSON.stringify({ ...(row.id !== undefined ? { id: row.id } : {}), ...await classifier.evaluate({ state }) }));
        }
      } else {
        const state = stateText(string('text') ?? JSON.parse(requireString('state'))).state;
        print(await classifier.evaluate({ state }));
      }
    } finally { await classifier.dispose(); }
  } else if (command === 'evaluate') {
    const result = await evaluateBundle(requireString('model'), requireString('data'));
    print({ model: result.model, summary: result.summary, thresholdRecommendation: result.thresholdRecommendation, batchInferenceMs: result.batchInferenceMs, note: result.note });
  } else if (command === 'inspect') {
    const directory = resolve(requireString('model'));
    const manifest = await readManifest(directory);
    print({ model: manifest.id, task: manifest.task, backend: manifest.features.kind, thresholdRecommendation: manifest.thresholdRecommendation,
      report: JSON.parse(await readFile(join(directory, 'report.json'), 'utf8')) });
  } else if (command === 'prepare-browser') {
    const destination = resolve(requireString('out'));
    await prepareBrowser(destination);
    print({ directory: destination });
  } else if (command === 'prepare-encoder') {
    const destination = resolve(requireString('out'));
    let exists = false;
    try { await stat(destination); exists = true; }
    catch (error) { if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error; }
    if (exists) throw new Error(`Output already exists: ${destination}.`);
    await mkdir(dirname(destination), { recursive: true });
    const staging = await mkdtemp(join(dirname(destination), `.${basename(destination)}-`));
    try {
      console.error('Downloading public pretrained MiniLM assets…');
      const config = await prepareEncoder(staging);
      await rename(staging, destination);
      print({ ...config, directory: destination });
    } catch (error) { await rm(staging, { recursive: true, force: true }); throw error; }
  }
}
main().catch(error => { console.error(`Error: ${(error as Error).message}`); process.exitCode = 1; });

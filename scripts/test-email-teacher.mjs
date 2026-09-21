// Run after npm run build and generate-email-dataset.mjs. Only synthetic email inputs are uploaded.
import { readFile, writeFile, stat } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { spawn } from 'node:child_process';
import { parseArgs } from 'node:util';
import { AutoTokenizer } from '@huggingface/transformers';
import { readTeacherInputs, labelInputs } from '../dist/teacher.js';
import { parseTask } from '../dist/schema.js';
import { measureEmailModel } from './measure-email-model.mjs';

const { values } = parseArgs({ options: {
  dataset: { type: 'string', default: 'datasets/email-300' },
  out: { type: 'string', default: 'models/email-300/minilm' },
  encoder: { type: 'string', default: '.cache/minilm' },
  teacher: { type: 'string', default: 'typesafe-ai/jev' },
  'teacher-rpm': { type: 'string', default: '20' },
  report: { type: 'string', default: 'benchmarks/email-300.json' },
} });
const dataset = resolve(values.dataset), out = resolve(values.out), encoder = resolve(values.encoder);
try { await stat(out); throw new Error('Model output exists; choose another --out directory.'); }
catch (error) { if (error.code !== 'ENOENT') throw error; }
const taskPath = join(dataset, 'task.json');
const task = parseTask(JSON.parse(await readFile(taskPath, 'utf8')));
const all = await readTeacherInputs(join(dataset, 'inputs.jsonl'), task);
const test = await readTeacherInputs(join(dataset, 'test.inputs.jsonl'), task);
const training = await readTeacherInputs(join(dataset, 'train.inputs.jsonl'), task);
if (all.inputs.length !== 300 || training.inputs.length !== 240 || test.inputs.length !== 60) throw new Error('Expected the 300-email dataset with its preselected 240/60 split.');
const trainIds = new Set(training.inputs.map(row => row.id));
if (test.inputs.some(row => trainIds.has(row.id))) throw new Error('Test IDs overlap training IDs.');
// Fail before paid labeling if any email would exceed the unchanged local model limit.
const tokenizer = await AutoTokenizer.from_pretrained(encoder, { local_files_only: true });
const tokens = all.inputs.map(row => tokenizer(row.text, { truncation: false, padding: false, return_tensor: false }).input_ids.length);
if (Math.max(...tokens) > 256) throw new Error('An email exceeds MiniLM’s 256-wordpiece limit.');
const teacherCache = join(dataset, 'jev.teacher');
const options = { teacher: values.teacher, teacherCache, teacherRpm: Number(values['teacher-rpm']), onProgress: console.error };
const labelingStarted = performance.now();
const testLabels = await labelInputs(test.inputs, task, options);
const testPath = join(dataset, 'test.jsonl');
const jsonl = rows => rows.map(row => JSON.stringify(row)).join('\n') + '\n';
await writeFile(testPath, jsonl(test.inputs.map((row, i) => ({ id: row.id, state: row.state, response: testLabels.rows[i].response }))));

// Exercise the actual CLI for all 240 training/development inputs, with the same reusable cache.
const cliStarted = performance.now();
const cliResult = await new Promise((accept, reject) => {
  const child = spawn(process.execPath, ['dist/cli.js', 'train', '--task', taskPath,
    '--inputs', join(dataset, 'train.inputs.jsonl'), '--teacher', values.teacher,
    '--teacher-cache', teacherCache, '--teacher-rpm', values['teacher-rpm'], '--test', testPath, '--encoder', encoder, '--out', out],
  { stdio: ['ignore', 'pipe', 'inherit'] });
  let stdout = '';
  child.stdout.setEncoding('utf8').on('data', chunk => { stdout += chunk; });
  child.on('error', reject);
  child.on('close', code => {
    if (code !== 0) reject(new Error(`Training CLI exited with ${code}. Labels remain cached.`));
    else { try { accept(JSON.parse(stdout)); } catch (error) { reject(error); } }
  });
});
const cliElapsedMs = performance.now() - cliStarted;
const endToEndMs = performance.now() - labelingStarted;
// Export all 300 joined labels for reuse; this must make no additional requests.
const exported = await labelInputs(all.inputs, task, options);
if (exported.summary.requests !== 0) throw new Error('Unexpected cache miss when exporting the complete dataset.');
await writeFile(join(dataset, 'outputs.jsonl'), jsonl(exported.rows));
const result = await measureEmailModel({ dataset, out, reportFile: values.report,
  run: { cli: cliResult, cliElapsedMs, endToEndMs, testLabeling: testLabels.summary } });
console.log(JSON.stringify({ report: values.report, summary: result.local.summary, single: result.local.single.accuracy,
  inference: result.local.inference, batchVsSingle: result.local.batchVsSingle }, null, 2));

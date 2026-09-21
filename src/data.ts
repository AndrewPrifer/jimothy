import { readFile } from 'node:fs/promises';
import { canonical, json, labelIndex, object, oneHot, parseTask, sameTask, stateText, targetFromAnswer, digest } from './schema.js';
import type { Example, Task } from './types.js';

export async function readRows(path: string): Promise<unknown[]> {
  const text = (await readFile(path, 'utf8')).replace(/^\uFEFF/, '').trim();
  if (!text) throw new Error(`${path} is empty.`);
  if (text.startsWith('[')) {
    const rows: unknown = JSON.parse(text);
    if (!Array.isArray(rows) || rows.length === 0) throw new Error(`${path} must contain a non-empty JSON array.`);
    return rows;
  }
  return text.split(/\r?\n/).filter(line => line.trim()).map((line, i) => {
    try { return JSON.parse(line) as unknown; }
    catch { throw new Error(`${path}:${i + 1}: invalid JSON. Use JSONL or a JSON array.`); }
  });
}
function optionalString(value: unknown, name: string): string | undefined {
  if (value === undefined) return undefined;
  if (typeof value !== 'string' || !value.trim()) throw new Error(`${name} must be a non-empty string.`);
  return value;
}
export async function joinFiles(inputs: string, outputs: string): Promise<unknown[]> {
  const [inputRows, outputRows] = await Promise.all([readRows(inputs), readRows(outputs)]);
  const map = new Map<string, Record<string, unknown>>();
  for (const row of outputRows) {
    const out = object(row, 'Output row');
    const id = optionalString(out.id, 'Output id');
    if (!id) throw new Error('Separate files require an id on every row; positional joins are not supported.');
    if (map.has(id)) throw new Error(`Duplicate output id: ${id}.`);
    map.set(id, out);
  }
  const seen = new Set<string>();
  const rows = inputRows.map(row => {
    const input = object(row, 'Input row');
    const id = optionalString(input.id, 'Input id');
    if (!id || !map.has(id)) throw new Error(`Missing output for input id ${String(id)}.`);
    if (seen.has(id)) throw new Error(`Duplicate input id: ${id}.`);
    seen.add(id);
    const out = map.get(id)!;
    return { ...input, output: out.output ?? out.response ?? out };
  });
  if (seen.size !== map.size) throw new Error('Output file contains ids with no matching input.');
  return rows;
}
export function parseDataset(rows: unknown[], initialTask?: Task, questionId?: string): { task: Task; examples: Example[]; duplicatesRemoved: number } {
  let task = initialTask;
  const examples: Example[] = [];
  const ids = new Set<string>();
  const texts = new Map<string, Example>();
  let duplicatesRemoved = 0;
  for (let i = 0; i < rows.length; i++) {
    try {
      const row = object(rows[i], 'Dataset row');
      const rawInput = row.request ?? row.input;
      const request = rawInput && typeof rawInput === 'object' && !Array.isArray(rawInput) ? object(rawInput, 'request') : row;
      if (request.questions !== undefined) {
        const rowTask = parseTask({ questions: request.questions }, questionId ?? task?.questionId);
        if (task && !sameTask(task, rowTask)) throw new Error('Question instructions, type, or criteria changed within this dataset. Train a separate task version.');
        task = rowTask;
      }
      if (!task) throw new Error('Provide --task or include questions in the first request.');
      const { state, text } = stateText(request.state ?? (typeof rawInput === 'string' ? rawInput : row.state));
      const id = optionalString(row.id, 'id');
      if (id && ids.has(id)) throw new Error(`Duplicate id: ${id}.`);
      if (id) ids.add(id);
      const group = optionalString(row.group, 'group');
      const rawOutput = row.response ?? row.output;
      const output = rawOutput === undefined ? row : rawOutput;
      const response = output && typeof output === 'object' && !Array.isArray(output) ? object(output, 'response') : undefined;
      const answers = response?.answers === undefined ? undefined : object(response.answers, 'answers');
      const answer = answers ? answers[task.questionId] : rawOutput;
      const teacher = answer === undefined ? undefined : targetFromAnswer(answer, task);
      const humanLabel = row.label === undefined ? undefined : labelIndex(row.label, task);
      if (!teacher && humanLabel === undefined) throw new Error(`Missing answer for ${task.questionId}; provide answers or a human label.`);
      const target = humanLabel === undefined ? teacher! : oneHot(humanLabel, task.labels.length);
      const example: Example = { state, text, target, teacher, humanLabel, id, group,
        teacherModel: typeof response?.model === 'string' ? response.model : undefined };
      const previous = texts.get(text);
      if (previous) {
        if (canonical({ target: previous.target, group: previous.group ?? null, label: previous.humanLabel ?? null, teacher: previous.teacher ?? null }) !==
            canonical({ target, group: group ?? null, label: humanLabel ?? null, teacher: teacher ?? null })) {
          throw new Error('Identical inputs have conflicting targets or groups; resolve them before training.');
        }
        duplicatesRemoved++;
      } else {
        texts.set(text, example);
        examples.push(example);
      }
    } catch (error) { throw new Error(`Row ${i + 1}: ${(error as Error).message}`); }
  }
  if (!task || examples.length === 0) throw new Error('Dataset contains no examples.');
  return { task, examples, duplicatesRemoved };
}
export function argmax(values: ArrayLike<number>): number {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
  return best;
}
function groupedStrata(examples: Example[], seed: number): [string, Example[]][][] {
  const groups = new Map<string, Example[]>();
  for (const e of examples) {
    const key = e.group === undefined ? `text:${digest(e.text)}` : `group:${e.group}`;
    groups.set(key, [...(groups.get(key) ?? []), e]);
  }
  const strata = new Map<number, [string, Example[]][]>();
  for (const entry of groups) {
    const totals = entry[1][0].target.map((_, c) => entry[1].reduce((sum, e) => sum + e.target[c], 0));
    const label = argmax(totals);
    strata.set(label, [...(strata.get(label) ?? []), entry]);
  }
  const result = [...strata.entries()].sort((a, b) => a[0] - b[0]).map(([, entries]) => entries);
  for (const entries of result) {
    entries.sort((a, b) => digest(`${seed}:${a[0]}`).localeCompare(digest(`${seed}:${b[0]}`)));
    for (const [, rows] of entries) rows.sort((a, b) => a.text.localeCompare(b.text));
  }
  return result;
}
export function splitExamples(examples: Example[], seed: number): { train: Example[]; validation: Example[] } {
  const train: Example[] = [], validation: Example[] = [];
  for (const entries of groupedStrata(examples, seed)) {
    const count = entries.length < 2 ? 0 : Math.max(1, Math.floor(entries.length * 0.2));
    entries.forEach((entry, i) => (i < count ? validation : train).push(...entry[1]));
  }
  if (validation.length === 0) throw new Error('Cannot create a group-disjoint validation split. Provide more independent groups or --validation.');
  return { train, validation };
}
/** Disjoint model selection, calibration and acceptance-selection subsets. */
export function splitDevelopment(examples: Example[], seed: number) {
  const tuning: Example[] = [], calibration: Example[] = [], acceptance: Example[] = [];
  for (const entries of groupedStrata(examples, seed)) {
    // Keep scarce classes in model selection rather than reusing their rows downstream.
    const held = entries.length < 3 ? 0 : Math.max(1, Math.floor(entries.length / 4));
    entries.forEach(([, rows], i) => (i < held ? calibration : i < held * 2 ? acceptance : tuning).push(...rows));
  }
  return { tuning, calibration, acceptance };
}
export function assertDisjoint(a: Pick<Example, 'text' | 'group'>[], b: Pick<Example, 'text' | 'group'>[], name: string): void {
  const texts = new Set(a.map(e => e.text));
  const groups = new Set(a.flatMap(e => e.group === undefined ? [] : [e.group]));
  if (b.some(e => texts.has(e.text) || (e.group !== undefined && groups.has(e.group)))) {
    throw new Error(`${name} overlaps another split by input or group. Keep duplicate inputs and related groups in one split.`);
  }
}
export function datasetDigest(examples: Example[]): string {
  return digest(canonical(examples.map(e => json({ state: e.state, target: e.target, group: e.group ?? null }))));
}

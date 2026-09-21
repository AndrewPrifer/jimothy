import type { Json, Question, Task } from './types.js';

export function object(value: unknown, where: string): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error(`${where} must be an object.`);
  return value as Record<string, unknown>;
}
export function json(value: unknown, where = 'Value'): Json {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return value;
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (Array.isArray(value)) return value.map(v => json(v, where));
  if (value && typeof value === 'object') {
    const entries = Object.entries(value).map(([k, v]) => [k, json(v, where)]);
    return Object.fromEntries(entries);
  }
  throw new Error(`${where} must contain only finite JSON values.`);
}
export function canonical(value: unknown): string {
  const v = json(value);
  const sort = (x: Json): Json => Array.isArray(x) ? x.map(sort) : x && typeof x === 'object'
    ? Object.fromEntries(Object.keys(x).sort().map(k => [k, sort(x[k])])) : x;
  return JSON.stringify(sort(v));
}
export function stateText(value: unknown): { state: Json; text: string } {
  const state = json(value, 'state');
  if (state === null || (typeof state !== 'string' && !Array.isArray(state) && typeof state !== 'object')) {
    throw new Error('state must be a non-empty string, object, or array.');
  }
  const text = typeof state === 'string' ? state : canonical(state);
  if (!text.trim() || text === '{}' || text === '[]') throw new Error('state cannot be empty.');
  if (text.length > 100_000) throw new Error('state exceeds the 100,000-character safety limit; extract the relevant fields first.');
  return { state, text };
}
export function parseTask(value: unknown, questionId?: string): Task {
  const root = object(value, 'Task');
  const questions = root.questions !== undefined ? object(root.questions, 'questions') : root;
  const keys = Object.keys(questions);
  const id = questionId ?? (keys.length === 1 ? keys[0] : undefined);
  if (!id || !Object.hasOwn(questions, id)) throw new Error('Select a question with --question when the task has multiple questions.');
  const q = object(questions[id], `questions.${id}`);
  if (!['choice', 'boolean', 'noul', 'score'].includes(String(q.type))) throw new Error(`Unsupported question type: ${String(q.type)}.`);
  if (q.instructions === undefined) throw new Error(`Question ${id} needs instructions.`);
  const instructions = json(q.instructions, 'instructions');
  if (instructions === null || (typeof instructions === 'string' && !instructions.trim())) throw new Error('instructions cannot be empty.');
  const question: Question = { type: q.type as Question['type'], instructions };
  if (q.criteria !== undefined) question.criteria = json(q.criteria, 'criteria');
  let labels: string[];
  if (q.type === 'choice') {
    labels = Object.keys(object(q.criteria, 'Choice criteria')).sort();
    if (labels.length < 2 || labels.length > 255 || labels.some(x => !x.trim())) throw new Error('Choice needs 2–255 non-empty named criteria.');
  } else if (q.type === 'score') {
    if (!Array.isArray(q.criteria) || q.criteria.length < 2 || q.criteria.length > 10) throw new Error('Score needs 2–10 ordered criteria.');
    labels = q.criteria.map((_, i) => String(i));
  } else {
    labels = ['false', 'true'];
    if (q.criteria !== undefined) {
      const criteria = object(q.criteria, 'Boolean criteria');
      if (Object.keys(criteria).some(k => k !== 'true' && k !== 'false')) throw new Error('Boolean criteria keys must be true or false.');
    }
  }
  return { questionId: id, question, labels };
}
export function sameTask(a: Task, b: Task): boolean { return canonical(a) === canonical(b); }
export function probability(value: unknown, where: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0 || value > 1) throw new Error(`${where} must be a number between 0 and 1.`);
  return value;
}
export function labelIndex(value: unknown, task: Task): number {
  const label = typeof value === 'boolean' || (task.question.type === 'score' && Number.isInteger(value)) ? String(value) : value;
  const index = typeof label === 'string' ? task.labels.indexOf(label) : -1;
  if (index < 0) throw new Error(`Unknown label ${JSON.stringify(value)}. Expected one of ${task.labels.join(', ')}.`);
  return index;
}
export function oneHot(index: number, count: number): number[] {
  return Array.from({ length: count }, (_, i) => i === index ? 1 : 0);
}
export function targetFromAnswer(value: unknown, task: Task): number[] {
  if (typeof value === 'string' || typeof value === 'boolean') return oneHot(labelIndex(value, task), task.labels.length);
  const answer = object(value, 'Answer');
  const binary = task.question.type === 'noul' || task.question.type === 'boolean';
  if (answer.type !== undefined && answer.type !== task.question.type && !(binary && ['boolean', 'noul'].includes(String(answer.type)))) {
    throw new Error('Answer type does not match the selected question.');
  }
  if (binary) {
    const p = probability(answer.probability ?? answer.noul, 'Boolean probability/noul');
    return [1 - p, p];
  }
  if (answer.probabilities !== undefined) {
    const distribution = object(answer.probabilities, 'probabilities');
    if (Object.keys(distribution).length !== task.labels.length || task.labels.some(k => !Object.hasOwn(distribution, k))) {
      throw new Error('probabilities must contain exactly the task labels; partial distributions cannot be imported.');
    }
    const values = task.labels.map(k => probability(distribution[k], `probabilities.${k}`));
    const sum = values.reduce((a, b) => a + b, 0);
    if (Math.abs(sum - 1) > 0.01) throw new Error(`probabilities must sum to 1 (received ${sum}).`);
    // Tolerate rounding in logged responses while preserving their relative probabilities.
    return values.map(v => v / sum);
  }
  if (task.question.type === 'choice' && answer.choice !== undefined) return oneHot(labelIndex(answer.choice, task), task.labels.length);
  throw new Error('Answer needs a full probability distribution (or a hard choice label). Score means alone are insufficient.');
}

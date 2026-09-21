import { appendFile, mkdir, open, readFile, rename, rm, truncate, writeFile } from 'node:fs/promises';
import { hostname } from 'node:os';
import { join, resolve } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';
import { readRows } from './data.js';
import { canonical, digest, json, object, parseTask, sameTask, stateText, targetFromAnswer } from './schema.js';
import type { Json, Task } from './types.js';

export const DEFAULT_TEACHER_URL = 'https://ai-gateway.vercel.sh/typesafe/v1/systemone';
export interface TeacherInput { id: string; state: Json; text: string; group?: string }
export interface TeacherOptions {
  teacher: string;
  teacherUrl?: string;
  teacherKeyEnv?: string;
  teacherCache: string;
  teacherRpm?: number;
  onProgress?: (message: string) => void;
}

/** Validate every input before the first paid request. Labels are never silently ignored. */
export async function readTeacherInputs(path: string, initialTask?: Task, questionId?: string) {
  let task = initialTask;
  const inputs: TeacherInput[] = [];
  const ids = new Set<string>(), groups = new Map<string, string | undefined>();
  for (const [i, raw] of (await readRows(path)).entries()) {
    try {
      const row = object(raw, 'Input row');
      if (['label', 'output', 'response', 'answers'].some(key => row[key] !== undefined)) {
        throw new Error('--teacher expects unlabeled inputs; use --data or --outputs for existing labels.');
      }
      if (typeof row.id !== 'string' || !row.id.trim()) throw new Error('Every input needs a non-empty string id.');
      if (ids.has(row.id)) throw new Error('Duplicate input id.');
      ids.add(row.id);
      if (row.group !== undefined && (typeof row.group !== 'string' || !row.group.trim())) throw new Error('group must be a non-empty string.');
      const rawInput = row.request ?? row.input;
      const request = rawInput && typeof rawInput === 'object' && !Array.isArray(rawInput) ? object(rawInput, 'request') : row;
      if (request.questions !== undefined) {
        const embedded = parseTask({ questions: request.questions }, questionId ?? task?.questionId);
        if (task && !sameTask(task, embedded)) throw new Error('Question instructions, type, or criteria changed within this dataset.');
        task = embedded;
      }
      if (!task) throw new Error('Provide --task or include questions in the first request.');
      const { state, text } = stateText(request.state ?? (typeof rawInput === 'string' ? rawInput : row.state));
      const group = row.group as string | undefined;
      if (groups.has(text) && groups.get(text) !== group) throw new Error('Identical inputs have conflicting groups.');
      groups.set(text, group);
      inputs.push({ id: row.id, state, text, ...(group === undefined ? {} : { group }) });
    } catch (error) { throw new Error(`Row ${i + 1}: ${(error as Error).message}`); }
  }
  return { task: task!, inputs, uniqueCount: groups.size };
}

export function teacherEndpoint(options: Pick<TeacherOptions, 'teacher' | 'teacherUrl' | 'teacherKeyEnv' | 'teacherRpm'>): string {
  if (!options.teacher.trim()) throw new Error('--teacher must name a model.');
  if (options.teacherRpm !== undefined && (!Number.isInteger(options.teacherRpm) || options.teacherRpm < 1 || options.teacherRpm > 60_000)) {
    throw new Error('--teacher-rpm must be an integer between 1 and 60000.');
  }
  if (options.teacherKeyEnv !== undefined && !/^[A-Za-z_][A-Za-z0-9_]*$/.test(options.teacherKeyEnv)) {
    throw new Error('--teacher-key-env must be an environment variable name, not a key.');
  }
  let url: URL;
  try { url = new URL(options.teacherUrl ?? DEFAULT_TEACHER_URL); }
  catch { throw new Error('--teacher-url must be a valid URL.'); }
  if (url.username || url.password || url.search || url.hash ||
    !(url.protocol === 'https:' || (url.protocol === 'http:' && ['localhost', '127.0.0.1', '[::1]'].includes(url.hostname)))) {
    throw new Error('--teacher-url must use HTTPS (or loopback HTTP), without credentials, query parameters, or a fragment.');
  }
  return url.href;
}

function validateResponse(raw: unknown, task: Task): Record<string, Json> {
  const response = object(raw, 'Teacher response');
  if (typeof response.model !== 'string' || !response.model.trim()) throw new Error('Teacher response needs a model identifier.');
  const answers = object(response.answers, 'Teacher answers');
  if (!Object.hasOwn(answers, task.questionId)) throw new Error('Teacher response is missing the selected answer.');
  targetFromAnswer(answers[task.questionId], task);
  return object(json(response), 'Teacher response') as Record<string, Json>;
}

async function acquireLock(directory: string): Promise<() => Promise<void>> {
  const path = join(directory, 'lock.json');
  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      const handle = await open(path, 'wx', 0o600);
      try { await handle.writeFile(JSON.stringify({ pid: process.pid, hostname: hostname() })); }
      finally { await handle.close(); }
      return () => rm(path, { force: true });
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== 'EEXIST') throw error;
      let stale = false;
      try {
        const lock = JSON.parse(await readFile(path, 'utf8'));
        if (lock.hostname === hostname() && Number.isInteger(lock.pid) && lock.pid > 0) {
          try { process.kill(lock.pid, 0); }
          catch (cause) { stale = (cause as NodeJS.ErrnoException).code === 'ESRCH'; }
        }
      } catch { /* An unreadable or foreign lock must not be taken over. */ }
      if (!stale || attempt) throw new Error('Teacher cache is locked by another run. Use a different cache or remove lock.json after confirming that run has stopped.');
      await rm(path);
    }
  }
  throw new Error('Unable to lock teacher cache.');
}

async function atomicJson(path: string, content: string): Promise<void> {
  const temporary = `${path}.${process.pid}.tmp`;
  await writeFile(temporary, content, { mode: 0o600 });
  await rename(temporary, path);
}

/** The cache is task/model/endpoint scoped; input subsets can share completed state responses. */
export async function labelInputs(inputs: TeacherInput[], task: Task, options: TeacherOptions) {
  const endpoint = teacherEndpoint(options);
  const directory = resolve(options.teacherCache);
  const identity = { format: 'jev-distill-teacher', version: 1, endpoint, model: options.teacher, task };
  const identityHash = digest(canonical(identity));
  const started = performance.now();
  await mkdir(directory, { recursive: true, mode: 0o700 });
  const unlock = await acquireLock(directory);
  try {
    const metadataPath = join(directory, 'run.json');
    let metadata: unknown;
    try { metadata = JSON.parse(await readFile(metadataPath, 'utf8')); }
    catch (error) { if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw new Error('Invalid teacher cache metadata. Use a new cache directory.'); }
    if (metadata !== undefined && canonical(metadata) !== canonical(identity)) {
      throw new Error('Teacher cache task, model, or endpoint changed. Use a new --teacher-cache directory.');
    }
    if (metadata === undefined) await atomicJson(metadataPath, JSON.stringify(identity, null, 2) + '\n');
    const responsePath = join(directory, 'responses.jsonl');
    const cached = new Map<string, Record<string, Json>>();
    let cacheText = '';
    try { cacheText = await readFile(responsePath, 'utf8'); }
    catch (error) { if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error; }
    // A killed append can leave a partial final line. Only newline-terminated records are committed.
    if (cacheText && !cacheText.endsWith('\n')) {
      cacheText = cacheText.slice(0, cacheText.lastIndexOf('\n') + 1);
      await truncate(responsePath, Buffer.byteLength(cacheText));
    }
    for (const line of cacheText.split('\n').filter(Boolean)) {
      const entry = object(JSON.parse(line), 'Cached response');
      if (typeof entry.key !== 'string' || !/^[a-f0-9]{64}$/.test(entry.key) || cached.has(entry.key)) throw new Error('Invalid or duplicate teacher cache key.');
      cached.set(entry.key, validateResponse(entry.response, task));
    }
    const keyOf = (input: TeacherInput) => digest(canonical({ identityHash, state: input.state }));
    const unique = new Map(inputs.map(input => [keyOf(input), input]));
    const pending = [...unique].filter(([key]) => !cached.has(key));
    const keyEnv = options.teacherKeyEnv ?? 'AI_GATEWAY_API_KEY';
    const apiKey = process.env[keyEnv];
    if (pending.length && !apiKey?.trim()) throw new Error(`Set ${keyEnv} before labeling with --teacher. Completed cache entries can be reused without a key.`);
    let requests = 0, retries = 0, completed = 0, next = 0;
    let failure: Error | undefined;
    let append = Promise.resolve();
    let gate = Promise.resolve(), nextStart = 0, cooldownUntil = 0;
    const interval = options.teacherRpm === undefined ? 0 : 60_000 / options.teacherRpm;
    const acquireSlot = () => {
      const turn = gate.then(async () => {
        while (Math.max(nextStart, cooldownUntil) > Date.now()) {
          await delay(Math.max(nextStart, cooldownUntil) - Date.now());
        }
        if (failure) throw new Error('Teacher labeling stopped after an earlier failure.');
        nextStart = Date.now() + interval;
      });
      gate = turn.catch(() => {});
      return turn;
    };
    const request = async (input: TeacherInput) => {
      const question = { ...task.question, type: task.question.type === 'boolean' ? 'noul' : task.question.type };
      for (let attempt = 0; attempt < 3; attempt++) {
        await acquireSlot();
        let response: Response;
        requests++;
        try {
          response = await fetch(endpoint, {
            method: 'POST', redirect: 'error', signal: AbortSignal.timeout(30_000),
            headers: { authorization: `Bearer ${apiKey!}`, 'content-type': 'application/json' },
            body: JSON.stringify({ model: options.teacher, state: input.state, questions: { [task.questionId]: question } }),
          });
        } catch {
          // Never echo provider/network errors: they can include credentials or email content.
          if (attempt === 2) throw new Error('Teacher request failed or timed out after 3 attempts. Rerun to resume completed labels.');
          retries++;
          options.onProgress?.('Teacher: network failure or timeout; retrying…');
          await delay(500 * 2 ** attempt);
          continue;
        }
        if (!response.ok) {
          await response.body?.cancel();
          if ((response.status === 429 || response.status >= 500) && attempt < 2) {
            const after = response.headers.get('retry-after');
            const retryMs = after === null ? 0 : Number.isFinite(Number(after)) ? Number(after) * 1000 : Date.parse(after) - Date.now();
            if (retryMs > 60_000) throw new Error('Teacher requested a retry delay over 60 seconds. Rerun later to resume.');
            retries++;
            const waitMs = Math.max(500 * 2 ** attempt, Number.isFinite(retryMs) ? retryMs : 0);
            options.onProgress?.(`Teacher: HTTP ${response.status}; retrying in ${(waitMs / 1000).toFixed(1)} seconds…`);
            // A rate limit pauses new work as well as retries, avoiding a burst from other workers.
            if (response.status === 429) cooldownUntil = Math.max(cooldownUntil, Date.now() + waitMs);
            else await delay(waitMs);
            continue;
          }
          throw new Error(`Teacher returned HTTP ${response.status}. Completed labels are saved; fix the error and rerun to resume.`);
        }
        try {
          const raw: unknown = await response.json();
          // Retain the Jev-compatible response and usage, but never persist an echoed credential.
          return validateResponse(JSON.parse(JSON.stringify(raw).split(apiKey!).join('[REDACTED]')), task);
        } catch { throw new Error('Teacher returned invalid JSON or an answer that does not match the task. Completed labels are saved.'); }
      }
      throw new Error('Teacher request exhausted retries.');
    };
    options.onProgress?.(`Teacher: ${unique.size - pending.length} cached, ${pending.length} unique inputs to label…`);
    await Promise.all(Array.from({ length: Math.min(4, pending.length) }, async () => {
      while (!failure) {
        const item = pending[next++];
        if (!item) return;
        const [key, input] = item;
        try {
          const response = await request(input);
          append = append.then(() => appendFile(responsePath, JSON.stringify({ key, response }) + '\n', { mode: 0o600 }));
          await append;
          cached.set(key, response);
          completed++;
          if (completed % 10 === 0 || completed === pending.length) options.onProgress?.(`Teacher: labeled ${completed}/${pending.length} inputs…`);
        } catch (error) { failure ??= error as Error; }
      }
    }));
    if (failure) throw failure;
    const rows = inputs.map(input => ({ id: input.id, response: cached.get(keyOf(input))! }));
    const outputs = join(directory, 'outputs.jsonl');
    await atomicJson(outputs, rows.map(row => JSON.stringify(row)).join('\n') + '\n');
    const summary = { model: options.teacher, endpoint, cache: directory, identityHash,
      inputs: inputs.length, uniqueInputs: unique.size, cached: unique.size - pending.length,
      labeled: completed, requests, retries, elapsedMs: performance.now() - started,
      requestsPerMinute: options.teacherRpm ?? null,
      resolvedModels: [...new Set(rows.map(row => row.response.model))].sort() };
    return { outputs, rows, summary };
  } finally { await unlock(); }
}

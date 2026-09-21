#!/usr/bin/env node
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { dirname, resolve, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { browserAssets } from '../../dist/browser-assets.js';
import { readManifest } from '../../dist/artifact.js';
import { teacherEndpoint } from '../../dist/teacher.js';
import { createHandler } from './handler.mjs';
export { createHandler, SHARED_KEY_EXPIRES_AT } from './handler.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '../..');
export async function siteAssets(modelPath = resolve(root, process.env.EMAIL_MODEL_DIR ?? 'models/email-300/minilm')) {
  const manifest = await readManifest(modelPath);
  const expected = ['forum', 'primary', 'promotion', 'purchase', 'social', 'update'];
  if (manifest.features.kind !== 'minilm' || manifest.task.question.type !== 'choice' || JSON.stringify(manifest.task.labels) !== JSON.stringify(expected)) {
    throw new Error('Use the trained six-category MiniLM email model with EMAIL_MODEL_DIR.');
  }
  const teacher = process.env.EMAIL_TEACHER_MODEL ?? 'typesafe-ai/jev';
  const endpoint = teacherEndpoint({ teacher, teacherUrl: process.env.EMAIL_TEACHER_URL });
  const browser = await browserAssets();
  const versions = browser.versions;
  const fingerprint = createHash('sha256').update(JSON.stringify({ manifest, versions, worker: createHash('sha256').update(await readFile(join(root, 'dist/browser-worker.js'))).digest('hex') })).digest('hex').slice(0, 24);
  const base = `/assets/${fingerprint}/`;
  const config = { base, assetsUrl: `${base}sdk/`, labels: manifest.task.labels, questionId: manifest.task.questionId, modelId: manifest.id,
    teacherModel: teacher, maxTokens: manifest.features.maxTokens,
    policies: { wasm: { calibration: manifest.calibration, thresholdRecommendation: manifest.thresholdRecommendation }, webgpu: null } };
  const files = new Map([
    ['/', join(here, 'index.html')], ['/style.css', join(here, 'style.css')], ['/app.js', join(here, 'app.js')], ['/email.js', join(here, 'email.js')],
    ['/file-help.js', join(here, 'file-help.js')], ['/landing.js', join(here, 'landing.js')],
    ['/jimothy/browser.js', join(root, 'dist/browser.js')], [`${base}model.json`, join(modelPath, 'model.json')],
  ]);
  for (const file of Object.keys(manifest.files)) files.set(`${base}${file}`, join(modelPath, file));
  for (const [name, source] of browser.files) files.set(`${config.assetsUrl}${name}`, source);
  return { manifest, config, files, teacher, endpoint };
}

export async function start() {
  const { manifest, config, files, teacher, endpoint } = await siteAssets();
  const apiKey = process.env.AI_GATEWAY_API_KEY?.trim();
  const port = Number(process.env.EMAIL_DEMO_PORT ?? 4320);
  if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error('EMAIL_DEMO_PORT must be a valid port number.');
  const server = createServer(createHandler({ manifest, config, files, apiKey, teacher, endpoint, port: () => port }));
  server.on('error', error => { console.error(`Cannot start email playground: ${error.message}`); process.exitCode = 1; });
  server.listen(port, '127.0.0.1', () => console.log(`Email playground: http://127.0.0.1:${port}\nJev: ${apiKey ? 'ready' : 'set AI_GATEWAY_API_KEY to enable'}\nPress Ctrl-C to stop.`));
  return server;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  start().catch(error => { console.error(`Cannot start email playground: ${error.message}`); process.exitCode = 1; });
}

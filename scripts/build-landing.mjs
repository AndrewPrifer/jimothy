import { copyFile, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { dirname, resolve, join } from 'node:path';
import { createHash } from 'node:crypto';
import { build } from 'esbuild';
import { siteAssets } from '../examples/email-browser/server.mjs';

const root = resolve(import.meta.dirname, '..');
const output = join(root, '.vercel/output');
const modelPath = join(root, 'examples/email-browser/model');
const { manifest, config, files } = await siteAssets(modelPath);
// Verify the checked-in bundle before publishing any of its assets.
for (const [file, expected] of Object.entries(manifest.files)) {
  const actual = createHash('sha256').update(await readFile(join(modelPath, file))).digest('hex');
  if (actual !== expected) throw new Error('Email model checksum mismatch: ' + file);
}
await rm(output, { recursive: true, force: true });
for (const [url, source] of files) {
  const destination = join(output, 'static', url === '/' ? 'index.html' : url.slice(1));
  await mkdir(dirname(destination), { recursive: true });
  await copyFile(source, destination);
}
const func = join(output, 'functions/api/jev.func');
await mkdir(func, { recursive: true });
// Only task/config and validation code go into the function, never the encoder.
await build({
  stdin: {
    contents: `import { createHandler } from './examples/email-browser/handler.mjs';
import { teacherEndpoint } from './dist/teacher.js';
const teacher = process.env.EMAIL_TEACHER_MODEL || 'typesafe-ai/jev';
export default createHandler({
  manifest: ${JSON.stringify({task: manifest.task})},
  config: { ...${JSON.stringify(config)}, teacherModel: teacher },
  files: new Map(), deployed: true,
  apiKey: process.env.AI_GATEWAY_API_KEY?.trim(),
  teacher, endpoint: teacherEndpoint({teacher, teacherUrl: process.env.EMAIL_TEACHER_URL})
});`,
    resolveDir: root, sourcefile: 'landing-function.mjs',
  },
  bundle: true, platform: 'node', format: 'esm', target: 'node22',
  outfile: join(func, 'index.mjs'),
});
await writeFile(join(func, '.vc-config.json'), JSON.stringify({
  runtime: 'nodejs22.x', handler: 'index.mjs', launcherType: 'Nodejs', maxDuration: 60,
}));
const configFunc = join(output, 'functions/config.json.func');
await mkdir(configFunc, { recursive: true });
await copyFile(join(func, 'index.mjs'), join(configFunc, 'handler.mjs'));
await writeFile(join(configFunc, 'index.mjs'), `import handler from './handler.mjs';
export default function(req, res) { req.url = '/config.json'; return handler(req, res); }
`);
await copyFile(join(func, '.vc-config.json'), join(configFunc, '.vc-config.json'));
await writeFile(join(output, 'config.json'), JSON.stringify({
  version: 3,
  routes: [
    { src: '/(.*)', headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'X-Content-Type-Options': 'nosniff',
      'Content-Security-Policy': "default-src 'self'; script-src 'self' 'wasm-unsafe-eval' blob:; style-src 'self'; worker-src 'self' blob:; connect-src 'self' blob:; img-src 'self' data:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
    }, continue: true },
    { src: '/assets/(.*)', headers: { 'Cache-Control': 'public, max-age=31536000, immutable' }, continue: true },
    { handle: 'filesystem' },
  ],
}, null, 2));
console.log('Landing page built in .vercel/output');

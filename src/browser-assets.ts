import { createRequire } from 'node:module';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { copyFile, lstat, mkdir, mkdtemp, readFile, rename, rm, writeFile } from 'node:fs/promises';

/** Asset map also used by the development examples; does not fetch anything. */
export async function browserAssets(): Promise<{ files: Map<string, string>; versions: Record<string, string> }> {
  const dist = fileURLToPath(new URL('../dist/', import.meta.url));
  let transformers: string, ort: string;
  try {
    const entry = import.meta.resolve('@huggingface/transformers');
    transformers = dirname(dirname(fileURLToPath(entry)));
    ort = dirname(dirname(createRequire(entry).resolve('onnxruntime-web')));
  } catch { throw new Error('prepare-browser requires @huggingface/transformers. Install jimothy with optional dependencies enabled.'); }
  const versions = Object.fromEntries(await Promise.all([[transformers, '@huggingface/transformers'], [ort, 'onnxruntime-web']].map(async ([directory, name]) =>
    [name, JSON.parse(await readFile(join(directory, 'package.json'), 'utf8')).version])));
  const files = new Map([
    ['browser.js', join(dist, 'browser.js')], ['browser-worker.js', join(dist, 'browser-worker.js')],
    ['runtime/transformers.min.js', join(transformers, 'dist/transformers.min.js')],
    ['runtime/transformers.LICENSE', join(transformers, 'LICENSE')], ['runtime/onnxruntime.LICENSE', join(dist, '../docs/licenses/onnxruntime.txt')],
  ]);
  for (const flavor of ['', '.jsep', '.asyncify', '.jspi']) for (const extension of ['mjs', 'wasm']) {
    const file = `ort-wasm-simd-threaded${flavor}.${extension}`;
    files.set(`runtime/${file}`, join(ort, 'dist', file));
  }
  return { files, versions };
}

export async function prepareBrowser(directory: string): Promise<void> {
  const destination = resolve(directory);
  try { await lstat(destination); throw new Error(`Output already exists: ${destination}.`); }
  catch (error) { if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error; }
  const { files, versions } = await browserAssets();
  await mkdir(dirname(destination), { recursive: true });
  const staging = await mkdtemp(join(dirname(destination), '.jimothy-browser-'));
  try {
    for (const [file, source] of files) {
      await mkdir(dirname(join(staging, file)), { recursive: true });
      await copyFile(source, join(staging, file));
    }
    await writeFile(join(staging, 'versions.json'), JSON.stringify(versions));
    await rename(staging, destination);
  } catch (error) { await rm(staging, { recursive: true, force: true }); throw error; }
}

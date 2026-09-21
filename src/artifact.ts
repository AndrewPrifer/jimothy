import { readFile, writeFile, lstat } from 'node:fs/promises';
import { join } from 'node:path';
import { digest } from './schema.js';
import { ENCODER_FILES, modelIdentity, parseManifest } from './manifest.js';
import type { Manifest } from './types.js';

function modelId(manifest: Omit<Manifest, 'id'> | Manifest): string {
  return `local-${digest(modelIdentity(manifest)).slice(0, 20)}`;
}
export async function writeManifest(directory: string, manifest: Omit<Manifest, 'id' | 'files'>): Promise<Manifest> {
  const filenames = ['report.json', ...(manifest.features.kind === 'minilm' ? ENCODER_FILES.map(f => `encoder/${f}`) : [])];
  const files: Record<string, string> = {};
  for (const file of filenames) files[file] = digest(await readFile(join(directory, file)));
  const base = { ...manifest, files };
  const result: Manifest = { ...base, id: modelId(base) };
  await writeFile(join(directory, 'model.json'), JSON.stringify(result));
  return result;
}
export async function readManifest(directory: string): Promise<Manifest> {
  const manifest = parseManifest(JSON.parse(await readFile(join(directory, 'model.json'), 'utf8')));
  for (const [file, hash] of Object.entries(manifest.files)) {
    if (!(await lstat(join(directory, file))).isFile()) throw new Error(`Bundle asset must be a regular file: ${file}.`);
    if (digest(await readFile(join(directory, file))) !== hash) throw new Error(`Checksum mismatch for ${file}; model bundle is damaged.`);
  }
  if (manifest.id !== modelId(manifest)) throw new Error('Model manifest checksum mismatch.');
  return manifest;
}

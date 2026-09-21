import { resolve } from 'node:path';
import { readManifest } from './artifact.js';
import { minilmExtractor } from './minilm.js';
import { tfidfExtractor } from './tfidf.js';
import { LocalClassifier } from './classifier.js';
export { LocalClassifier } from './classifier.js';

/** Loads and verifies a complete local bundle. Network access is never enabled. */
export async function loadClassifier(directory: string): Promise<LocalClassifier> {
  const location = resolve(directory);
  const manifest = await readManifest(location);
  const extractor = manifest.features.kind === 'tfidf' ? tfidfExtractor(manifest.features) : await minilmExtractor(manifest.features, location);
  return new LocalClassifier(manifest, extractor);
}

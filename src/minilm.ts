import { mkdir, readFile, writeFile, copyFile } from 'node:fs/promises';
import { resolve, dirname, join } from 'node:path';
import type { FeatureExtractor, MiniLMConfig } from './types.js';
import { encoderExtractor } from './encoder-extractor.js';

const MODEL_ID = 'Xenova/all-MiniLM-L6-v2';
import { ENCODER_FILES } from './manifest.js';
export { ENCODER_FILES } from './manifest.js';
const REQUIRED_FILES = ENCODER_FILES.slice(0, 5);

async function download(url: string, path: string): Promise<void> {
  const response = await fetch(url, { signal: AbortSignal.timeout(120_000) });
  if (!response.ok) throw new Error(`Encoder download failed: HTTP ${response.status} from ${url}.`);
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, new Uint8Array(await response.arrayBuffer()));
}
/** Download only public pretrained assets. No dataset is sent over the network. */
export async function prepareEncoder(directory: string, source?: string): Promise<MiniLMConfig> {
  await mkdir(directory, { recursive: true });
  let revision: string;
  if (source) {
    if (resolve(source) === resolve(directory)) throw new Error('Encoder source and destination must differ.');
    const metadata = JSON.parse(await readFile(join(source, 'encoder-source.json'), 'utf8'));
    if (metadata.modelId !== MODEL_ID || typeof metadata.revision !== 'string' || !/^[a-f0-9]{40}$/.test(metadata.revision)) {
      throw new Error('Use an encoder directory created by prepare-encoder; only the bundled MiniLM architecture is supported.');
    }
    revision = metadata.revision;
    for (const file of ENCODER_FILES) {
      await mkdir(dirname(join(directory, file)), { recursive: true });
      await copyFile(join(source, file), join(directory, file));
    }
  } else {
    const response = await fetch(`https://huggingface.co/api/models/${MODEL_ID}`, { signal: AbortSignal.timeout(30_000) });
    if (!response.ok) throw new Error(`Cannot resolve encoder revision: HTTP ${response.status}.`);
    const metadata = await response.json() as { sha?: unknown };
    if (typeof metadata.sha !== 'string' || !/^[a-f0-9]{40}$/.test(metadata.sha)) throw new Error('Encoder registry did not return an immutable revision.');
    revision = metadata.sha;
    for (const file of [...REQUIRED_FILES, 'README.md']) {
      await download(`https://huggingface.co/${MODEL_ID}/resolve/${revision}/${file}`, join(directory, file));
    }
    await download('https://www.apache.org/licenses/LICENSE-2.0.txt', join(directory, 'LICENSE.txt'));
    await writeFile(join(directory, 'encoder-source.json'), JSON.stringify({ modelId: MODEL_ID, revision }, null, 2));
  }
  return { kind: 'minilm', dimensions: 384, maxTokens: 256, directory: 'encoder', modelId: MODEL_ID, revision, dtype: 'q8' };
}
export async function minilmExtractor(config: MiniLMConfig, bundleDirectory: string): Promise<FeatureExtractor> {
  let transformers: typeof import('@huggingface/transformers');
  try { transformers = await import('@huggingface/transformers'); }
  catch { throw new Error('MiniLM requires @huggingface/transformers. Install optional dependencies, or use --backend tfidf.'); }
  const pipe = await transformers.pipeline('feature-extraction', resolve(bundleDirectory, config.directory), {
    dtype: 'q8', device: 'cpu', local_files_only: true,
    session_options: { intraOpNumThreads: 1, interOpNumThreads: 1 },
  });
  return encoderExtractor(config, pipe);
}

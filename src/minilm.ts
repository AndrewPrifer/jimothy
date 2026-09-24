import { mkdir, readFile, writeFile, copyFile } from 'node:fs/promises';
import { resolve, dirname, join } from 'node:path';
import type { FeatureExtractor, MiniLMConfig, Vector } from './types.js';

export const DEFAULT_MODEL_ID = 'Xenova/all-MiniLM-L6-v2';
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
export async function prepareEncoder(directory: string, source?: string, requestedModelId?: string): Promise<MiniLMConfig> {
  await mkdir(directory, { recursive: true });
  const modelId = requestedModelId ?? DEFAULT_MODEL_ID;
  if (!/^[A-Za-z0-9][\w.-]*\/[A-Za-z0-9][\w.-]*$/.test(modelId)) throw new Error('Encoder model must be a Hugging Face owner/model ID.');
  let revision: string;
  let actualModelId = modelId;
  if (source) {
    if (resolve(source) === resolve(directory)) throw new Error('Encoder source and destination must differ.');
    const metadata = JSON.parse(await readFile(join(source, 'encoder-source.json'), 'utf8'));
    if (typeof metadata.modelId !== 'string' || !/^[A-Za-z0-9][\w.-]*\/[A-Za-z0-9][\w.-]*$/.test(metadata.modelId) ||
        typeof metadata.revision !== 'string' || !/^[a-f0-9]{40}$/.test(metadata.revision)) throw new Error('Use an encoder directory created by prepare-encoder.');
    if (requestedModelId && requestedModelId !== metadata.modelId) throw new Error('--encoder-model does not match the local encoder.');
    actualModelId = metadata.modelId;
    revision = metadata.revision;
    for (const file of ENCODER_FILES) {
      await mkdir(dirname(join(directory, file)), { recursive: true });
      await copyFile(join(source, file), join(directory, file));
    }
  } else {
    const response = await fetch(`https://huggingface.co/api/models/${modelId}`, { signal: AbortSignal.timeout(30_000) });
    if (!response.ok) throw new Error(`Cannot resolve encoder revision: HTTP ${response.status}.`);
    const metadata = await response.json() as { sha?: unknown; cardData?: { license?: unknown }; siblings?: { rfilename?: string }[] };
    if (typeof metadata.sha !== 'string' || !/^[a-f0-9]{40}$/.test(metadata.sha)) throw new Error('Encoder registry did not return an immutable revision.');
    revision = metadata.sha;
    for (const file of [...REQUIRED_FILES, 'README.md']) {
      await download(`https://huggingface.co/${modelId}/resolve/${revision}/${file}`, join(directory, file));
    }
    const licenseFile = metadata.siblings?.find(file => file.rfilename === 'LICENSE' || file.rfilename === 'LICENSE.txt')?.rfilename;
    if (licenseFile) await download(`https://huggingface.co/${modelId}/resolve/${revision}/${licenseFile}`, join(directory, 'LICENSE.txt'));
    else if (modelId === DEFAULT_MODEL_ID) await download('https://www.apache.org/licenses/LICENSE-2.0.txt', join(directory, 'LICENSE.txt'));
    else await writeFile(join(directory, 'LICENSE.txt'), `No license file is provided by ${modelId} at ${revision}. Check https://huggingface.co/${modelId}/tree/${revision} before redistribution.\nDeclared license: ${typeof metadata.cardData?.license === 'string' ? metadata.cardData.license : 'unspecified'}\n`);
    await writeFile(join(directory, 'encoder-source.json'), JSON.stringify({ modelId, revision }, null, 2));
  }
  const architecture = JSON.parse(await readFile(join(directory, 'config.json'), 'utf8'));
  const tokenizer = JSON.parse(await readFile(join(directory, 'tokenizer_config.json'), 'utf8'));
  const dimensions = architecture.hidden_size;
  const maxTokens = actualModelId === DEFAULT_MODEL_ID ? 256 : Math.min(architecture.max_position_embeddings, tokenizer.model_max_length);
  if (!Number.isInteger(dimensions) || dimensions < 1 || dimensions > 100_000 ||
      !Number.isInteger(maxTokens) || maxTokens < 2 || maxTokens > 100_000) throw new Error('Encoder must declare valid hidden_size and token limits.');
  return { kind: 'minilm', dimensions, maxTokens, directory: 'encoder', modelId: actualModelId, revision, dtype: 'q8' };
}
export async function minilmExtractor(config: MiniLMConfig, bundleDirectory: string): Promise<FeatureExtractor> {
  let transformers: typeof import('@huggingface/transformers');
  try { transformers = await import('@huggingface/transformers'); }
  catch { throw new Error('The encoder requires @huggingface/transformers. Install optional dependencies, or use --backend tfidf.'); }
  const pipe = await transformers.pipeline('feature-extraction', resolve(bundleDirectory, config.directory), {
    dtype: 'q8', device: 'cpu', local_files_only: true,
    session_options: { intraOpNumThreads: 1, interOpNumThreads: 1 },
  });
  return {
    config,
    async encode(texts) {
      const inputs = texts.map(text => `${config.inputPrefix ?? ''}${text}`);
      // Check length before the pipeline can silently truncate. This includes special tokens.
      for (let i = 0; i < inputs.length; i++) {
        const tokenized = pipe.tokenizer(inputs[i], { truncation: false, padding: false, return_tensor: false });
        const ids = tokenized.input_ids as number[];
        if (ids.length > config.maxTokens) throw new Error(`Input ${i + 1} has ${ids.length} wordpieces; encoder supports ${config.maxTokens}. Extract relevant fields first.`);
      }
      const vectors: Vector[] = [];
      for (let start = 0; start < inputs.length; start += 32) {
        const result = await pipe(inputs.slice(start, start + 32), { pooling: 'mean', normalize: true });
        const rows = result.tolist() as number[][];
        for (const row of rows) {
          if (row.length !== config.dimensions || row.some(v => !Number.isFinite(v))) throw new Error('Encoder produced invalid embeddings.');
          vectors.push({ indices: Uint32Array.from(row, (_, i) => i), values: Float32Array.from(row) });
        }
      }
      return vectors;
    },
    async dispose() { await pipe.dispose(); },
  };
}

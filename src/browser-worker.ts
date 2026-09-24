import { LocalClassifier } from './classifier.js';
import { applyWebGPUCalibration, readBrowserBundle, sha256 } from './browser-bundle.js';
import { tfidfExtractor } from './tfidf.js';
import type { BrowserMetadata, BrowserProgress } from './browser.js';
import type { FeatureExtractor, MiniLMConfig, Vector } from './types.js';

const scope = globalThis as unknown as {
  postMessage(value: unknown): void;
  onmessage: (event: MessageEvent) => void;
  navigator: { gpu?: { requestAdapter(): Promise<{ features: Set<string> } | null> } };
};
let classifier: LocalClassifier | undefined;
const progress = (value: BrowserProgress) => scope.postMessage({ progress: value });
interface LoadArgs {
  manifestUrl: string; assetsUrl: string; device: 'wasm' | 'webgpu';
  webgpu?: { modelUrl: string; policyUrl: string };
  started: number; fallbackReason?: string;
}

async function load(args: LoadArgs): Promise<BrowserMetadata> {
  if (classifier) throw new Error('Classifier already loaded.');
  const { manifestUrl, assetsUrl, device } = args;
  progress({ phase: 'download', file: 'model.json', message: 'Loading model…' });
  const bundle = await readBrowserBundle(manifestUrl, file => progress({ phase: 'download', file, message: `Loading ${file}…` }));
  const { manifest, assets } = bundle;
  let extractor: FeatureExtractor;
  if (manifest.features.kind === 'tfidf') {
    if (device === 'webgpu') throw new Error('TF-IDF runs in JavaScript; WebGPU is supported for calibrated MiniLM exports.');
    extractor = tfidfExtractor(manifest.features);
  } else {
    const runtimeUrl = new URL('runtime/', assetsUrl);
    let adapter;
    if (device === 'webgpu') {
      adapter = await scope.navigator.gpu?.requestAdapter();
      if (!adapter?.features.has('shader-f16')) throw new Error('This browser does not provide an FP16-capable WebGPU adapter.');
      if (!args.webgpu) throw new Error('WebGPU requires a calibrated FP16 export.');
      const decode = (data: ArrayBuffer) => JSON.parse(new TextDecoder().decode(data));
      const runtime = decode(await bundle.download(new URL('versions.json', assetsUrl).href));
      const policy = decode(await bundle.download(args.webgpu.policyUrl));
      const hash = await applyWebGPUCalibration(manifest, policy, bundle.manifestSha256, runtime);
      progress({ phase: 'download', file: 'model_fp16.onnx', message: 'Loading FP16 encoder…' });
      const fp16 = await bundle.download(args.webgpu.modelUrl);
      if (await sha256(fp16) !== hash) throw new Error('Checksum mismatch for the WebGPU FP16 encoder.');
      assets.set(new URL('encoder/onnx/model_fp16.onnx', manifestUrl).href, fp16);
    }
    progress({ phase: 'initialize', message: `Starting ${device === 'wasm' ? 'WASM' : 'WebGPU'}…` });
    const moduleUrl = new URL('transformers.min.js', runtimeUrl).href;
    const { pipeline, env } = await import(/* @vite-ignore */ moduleUrl);
    env.allowRemoteModels = false; env.allowLocalModels = true;
    // Transformers' local-file discovery expects a path, not an absolute HTTP URL.
    env.localModelPath = new URL('.', manifestUrl).pathname;
    env.useBrowserCache = false; env.useWasmCache = false;
    env.backends.onnx.wasm.wasmPaths = runtimeUrl.href;
    env.backends.onnx.wasm.numThreads = 1; env.backends.onnx.wasm.proxy = false;
    if (adapter) env.backends.onnx.webgpu.adapter = adapter;
    // The encoder consumes the exact bytes that passed checksum verification.
    // Unlisted optional tokenizer/config files return 404 without any network request.
    env.fetch = async (url: string | URL) => {
      const data = assets.get(new URL(url, manifestUrl).href);
      return data ? new Response(data, { headers: { 'Content-Length': String(data.byteLength) } }) : new Response(null, { status: 404 });
    };
    const pipe = await pipeline('feature-extraction', 'encoder', { dtype: device === 'webgpu' ? 'fp16' : 'q8', device,
      local_files_only: true, session_options: { executionProviders: [device] } });
    if (typeof pipe.tokenizer !== 'function') throw new Error('The encoder tokenizer could not be loaded.');
    // Runtime loading is complete; avoid retaining a second copy of the encoder bytes.
    assets.clear();
    extractor = encoderExtractor(manifest.features, pipe);
  }
  classifier = new LocalClassifier(manifest, extractor);
  return { ...classifier.metadata, device: manifest.features.kind === 'tfidf' ? 'javascript' : device,
    dtype: manifest.features.kind === 'tfidf' ? null : device === 'webgpu' ? 'fp16' : 'q8',
    loadMs: performance.timeOrigin + performance.now() - args.started,
    ...(args.fallbackReason ? { fallbackReason: args.fallbackReason } : {}) };
}

function encoderExtractor(config: MiniLMConfig, pipe: any): FeatureExtractor {
  return { config,
    async encode(texts) {
      const inputs = texts.map(text => `${config.inputPrefix ?? ''}${text}`);
      for (let i = 0; i < inputs.length; i++) {
        const ids = pipe.tokenizer(inputs[i], { truncation: false, padding: false, return_tensor: false }).input_ids;
        if (ids.length > config.maxTokens) throw new Error(`Input ${i + 1} has ${ids.length} wordpieces; encoder supports ${config.maxTokens}. Extract relevant fields first.`);
      }
      const vectors: Vector[] = [];
      for (let start = 0; start < inputs.length; start += 32) {
        const rows: number[][] = (await pipe(inputs.slice(start, start + 32), { pooling: 'mean', normalize: true })).tolist();
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

// A failed request must not poison the queue, and ONNX sessions must not overlap.
let queue = Promise.resolve();
scope.onmessage = ({ data }) => {
  queue = queue.then(async () => {
    try {
      let result;
      if (data.method === 'load') result = await load(data.args);
      else if (data.method === 'predictBatch') {
        if (!classifier) throw new Error('Classifier has not loaded.');
        result = await classifier.predictBatch(data.args);
      } else throw new Error('Unknown browser worker method.');
      scope.postMessage({ id: data.id, result });
    } catch (error) { scope.postMessage({ id: data.id, error: error instanceof Error ? error.message : String(error) }); }
  });
};

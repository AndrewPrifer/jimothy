import type { LocalClassifier } from './classifier.js';
import type { Json, Prediction } from './types.js';
export type { Answer, Json, Prediction, Task, ThresholdRecommendation } from './types.js';

export interface BrowserProgress {
  phase: 'download' | 'initialize' | 'fallback';
  message: string;
  file?: string;
}
export interface BrowserOptions {
  /** Same-origin directory created by `jimothy prepare-browser`. Default: /jimothy/. */
  assetsUrl?: string | URL;
  /** WASM is the default. WebGPU requires a calibrated FP16 export. */
  device?: 'wasm' | 'webgpu' | 'auto';
  webgpu?: { modelUrl: string | URL; policyUrl: string | URL };
  /** Cancels loading. After loading, use dispose() to release the worker. */
  signal?: AbortSignal;
  onProgress?: (progress: BrowserProgress) => void;
}
export type BrowserMetadata = LocalClassifier['metadata'] & {
  device: 'wasm' | 'webgpu' | 'javascript';
  dtype: 'q8' | 'fp16' | null;
  loadMs: number;
  fallbackReason?: string;
};
type Pending = { resolve: (value: any) => void; reject: (error: Error) => void };

/** One isolated worker per classifier. Concurrent predictions are queued in order. */
export class BrowserClassifier {
  #worker: Worker;
  #pending = new Map<number, Pending>();
  #next = 0;
  #closed = false;
  #metadata?: BrowserMetadata;
  constructor(worker: Worker, onProgress?: BrowserOptions['onProgress']) {
    this.#worker = worker;
    worker.onmessage = ({ data }) => {
      if (data.progress) { try { onProgress?.(data.progress); } catch { /* Observers cannot break inference. */ } return; }
      const pending = this.#pending.get(data.id);
      if (!pending) return;
      this.#pending.delete(data.id);
      if (data.error) pending.reject(new Error(data.error));
      else pending.resolve(data.result);
    };
    worker.onerror = event => this.#close(new Error(event.message || 'Browser worker failed to load. Check assetsUrl and your Content Security Policy.'));
    worker.onmessageerror = () => this.#close(new Error('Cannot deserialize the browser worker response.'));
  }
  #close(error: Error) {
    this.#closed = true;
    this.#worker.terminate();
    for (const pending of this.#pending.values()) pending.reject(error);
    this.#pending.clear();
  }
  #request(method: string, args: unknown): Promise<any> {
    if (this.#closed) return Promise.reject(new Error('Classifier has been disposed.'));
    const id = ++this.#next;
    return new Promise((resolve, reject) => {
      this.#pending.set(id, { resolve, reject });
      try { this.#worker.postMessage({ id, method, args }); }
      catch (error) { this.#pending.delete(id); reject(error); }
    });
  }
  /** @internal Use loadClassifier(). */
  async initialize(args: unknown, signal?: AbortSignal) {
    const abort = () => this.#close(signal?.reason instanceof Error ? signal.reason : new DOMException('Model loading cancelled.', 'AbortError'));
    signal?.throwIfAborted();
    signal?.addEventListener('abort', abort, { once: true });
    try { this.#metadata = await this.#request('load', args); }
    finally { signal?.removeEventListener('abort', abort); }
  }
  get metadata(): BrowserMetadata {
    if (!this.#metadata) throw new Error('Classifier has not loaded.');
    return structuredClone(this.#metadata);
  }
  async predict(state: Json): Promise<Prediction> { return (await this.predictBatch([state]))[0]; }
  async predictBatch(states: Json[]): Promise<Prediction[]> { return this.#request('predictBatch', states); }
  async evaluate({ state }: { state: Json }) {
    const { answer } = await this.predict(state);
    return { model: this.metadata.id, answers: { [this.metadata.task.questionId]: answer } };
  }
  async dispose(): Promise<void> { this.#close(new Error('Classifier has been disposed.')); }
}

/** Load a complete model bundle from an HTTP(S) directory or model.json URL. */
export async function loadClassifier(modelUrl: string | URL, options: BrowserOptions = {}): Promise<BrowserClassifier> {
  if (typeof Worker === 'undefined' || typeof location === 'undefined') throw new Error('jimothy/browser requires a browser with module workers. Call loadClassifier on the client.');
  if (!globalThis.crypto?.subtle) throw new Error('Model verification requires HTTPS or localhost.');
  const url = (value: string | URL) => {
    const result = new URL(value, location.href);
    if (!['http:', 'https:'].includes(result.protocol) || result.username || result.password || result.search || result.hash) throw new Error('Use an HTTP(S) asset URL without credentials, query, or fragment.');
    return result;
  };
  const directory = (value: string | URL) => { const result = url(value); if (!result.pathname.endsWith('/')) result.pathname += '/'; return result; };
  const source = url(modelUrl);
  const manifestUrl = source.pathname.endsWith('.json') ? source : new URL('model.json', directory(source));
  const assetsUrl = directory(options.assetsUrl ?? '/jimothy/');
  if (assetsUrl.origin !== location.origin) throw new Error('assetsUrl must be on the same origin as the page (module worker requirement).');
  const device = options.device ?? 'wasm';
  if (!['wasm', 'webgpu', 'auto'].includes(device)) throw new Error('device must be wasm, webgpu, or auto.');
  if (device === 'webgpu' && !options.webgpu) throw new Error('WebGPU requires webgpu.modelUrl and webgpu.policyUrl from a calibrated FP16 export.');
  const webgpu = options.webgpu && { modelUrl: url(options.webgpu.modelUrl).href, policyUrl: url(options.webgpu.policyUrl).href };
  const started = performance.timeOrigin + performance.now();
  const load = async (backend: 'wasm' | 'webgpu', fallbackReason?: string) => {
    options.signal?.throwIfAborted();
    const classifier = new BrowserClassifier(new Worker(new URL('browser-worker.js', assetsUrl), { type: 'module', name: 'jimothy' }), options.onProgress);
    try {
      await classifier.initialize({ manifestUrl: manifestUrl.href, assetsUrl: assetsUrl.href, device: backend, webgpu, started, fallbackReason }, options.signal);
      return classifier;
    } catch (error) { await classifier.dispose(); throw error; }
  };
  if (device === 'wasm' || (device === 'auto' && !webgpu)) return load('wasm');
  try { return await load('webgpu'); }
  catch (error) {
    if (device !== 'auto' || options.signal?.aborted) throw error;
    const message = error instanceof Error ? error.message : String(error);
    try { options.onProgress?.({ phase: 'fallback', message }); } catch { /* Observer only. */ }
    // ONNX runtime state is global: a fresh worker keeps fallback independent of a failed GPU session.
    return load('wasm', message);
  }
}

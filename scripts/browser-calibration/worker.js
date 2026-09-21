import { headLogits, softmax } from '/linear.js';

const progress = value => postMessage({ progress: value });
async function get(url) {
  const response = await fetch(url), body = await response.json();
  if (!response.ok) throw new Error(body.error ?? `HTTP ${response.status}`);
  return body;
}
self.onmessage = async () => {
  let pipe;
  try {
    const config = await get('/config.json');
    const post = async (url, body) => {
      const response = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-Calibration-Token': config.token }, body: JSON.stringify(body) });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error ?? `HTTP ${response.status}`);
      return result;
    };
    const manifest = await get('/model.json');
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter || !adapter.features.has('shader-f16')) throw new Error('An FP16-capable WebGPU adapter is required.');
    const browser = { userAgent: navigator.userAgent, crossOriginIsolated, runtime: config.versions,
      adapter: { vendor: adapter.info?.vendor, architecture: adapter.info?.architecture, features: [...adapter.features].sort() } };
    const { pipeline, env } = await import('/vendor/transformers.min.js');
    env.allowRemoteModels = false; env.allowLocalModels = true; env.localModelPath = '/';
    env.useBrowserCache = false; env.useWasmCache = false;
    env.backends.onnx.wasm.wasmPaths = '/vendor/'; env.backends.onnx.wasm.numThreads = 1; env.backends.onnx.wasm.proxy = false;
    env.backends.onnx.webgpu.adapter = adapter;
    progress('Loading the pinned FP16 encoder on WebGPU…');
    pipe = await pipeline('feature-extraction', 'encoder', { dtype: 'fp16', device: 'webgpu', local_files_only: true,
      session_options: { executionProviders: ['webgpu'] } });
    const indices = Uint32Array.from({ length: manifest.features.dimensions }, (_, i) => i);
    async function encode(rows, batchSize, phase, policy) {
      const output = [];
      for (let start = 0; start < rows.length; start += batchSize) {
        if (start % (batchSize === 1 ? 200 : 320) === 0) progress(`${phase}: ${start}/${rows.length}`);
        const slice = rows.slice(start, start + batchSize);
        for (const row of slice) {
          if (pipe.tokenizer(row.text, { truncation: false, padding: false, return_tensor: false }).input_ids.length > manifest.features.maxTokens) throw new Error('Input exceeds token limit.');
        }
        const embeddings = (await pipe(slice.map(row => row.text), { pooling: 'mean', normalize: true })).tolist();
        embeddings.forEach((values, i) => {
          if (values.length !== indices.length || values.some(value => !Number.isFinite(value))) throw new Error('Invalid embedding.');
          const logits = headLogits(manifest.head, { indices, values: Float32Array.from(values) });
          const record = { id: slice[i].id, logits };
          if (policy) {
            const probabilities = softmax(logits, policy.calibration.temperature);
            const maxProbability = Math.max(...probabilities), winner = probabilities.indexOf(maxProbability);
            Object.assign(record, { choice: manifest.task.labels[winner], maxProbability,
              accepted: policy.acceptance.status === 'ready' && maxProbability >= policy.threshold });
          }
          output.push(record);
        });
      }
      return output;
    }
    const development = await get('/development.json');
    const calibration = await encode(development.calibration, 32, 'Calibration');
    const acceptance = await encode(development.acceptance, 32, 'Threshold selection');
    progress('Fitting and freezing the deployment policy…');
    const fitted = await post('/fit', { calibration, acceptance, browser });
    postMessage({ fitted });
    const { policy, policySha256 } = await get('/policy.json');
    if (policySha256 !== fitted.policySha256) throw new Error('Saved policy hash changed.');
    const test = await get('/test.json');
    const report = { fitted, evaluations: {} };
    for (const [mode, batchSize] of [['batch32', 32], ['single', 1]]) {
      const started = performance.now();
      const rows = await encode(test, batchSize, `Frozen policy test (${mode})`, policy);
      const evaluation = await post('/evaluate', { mode, policySha256, rows, elapsedMs: performance.now() - started });
      report.evaluations[mode] = evaluation;
      report.saved = evaluation.saved;
      postMessage(report);
    }
    await pipe.dispose(); pipe = undefined;
    postMessage({ ...report, complete: true });
  } catch (error) { postMessage({ error: String(error.message ?? error) }); }
  finally { if (pipe) await pipe.dispose().catch(() => {}); }
};

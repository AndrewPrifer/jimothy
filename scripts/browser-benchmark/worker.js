import { predictHead } from '/linear.js';

const progress = message => postMessage({ progress: message });
const stats = times => {
  const sorted = [...times].sort((a, b) => a - b);
  return { count: times.length, p50: sorted[Math.ceil(sorted.length * .5) - 1],
    p95: sorted[Math.ceil(sorted.length * .95) - 1], mean: times.reduce((a, b) => a + b, 0) / times.length, samplesMs: times };
};
self.onmessage = async ({ data: variant }) => {
  const result = { ...variant };
  let pipe;
  try {
    // Fetch fixture data outside timing. All states in this benchmark are plain strings.
    const [manifest, reference] = await Promise.all(['/model.json', '/reference.json'].map(async url => (await fetch(url)).json()));
    if (manifest.task.question.type !== 'choice' || reference.rows.some(row => typeof row.text !== 'string')) throw new Error('Benchmark requires string inputs and a choice task');
    let adapter;
    if (variant.device === 'webgpu') {
      adapter = await navigator.gpu?.requestAdapter();
      if (!adapter) throw new Error('No WebGPU adapter available');
      result.adapter = { vendor: adapter.info?.vendor, architecture: adapter.info?.architecture,
        device: adapter.info?.device, description: adapter.info?.description,
        features: [...adapter.features].sort(), isFallbackAdapter: adapter.info?.isFallbackAdapter ?? adapter.isFallbackAdapter };
    }
    const importStart = performance.now();
    const { pipeline, env } = await import('/vendor/transformers.min.js');
    result.runtimeImportMs = performance.now() - importStart;
    env.allowRemoteModels = false;
    env.allowLocalModels = true;
    env.localModelPath = '/';
    env.useBrowserCache = false;
    env.useWasmCache = false;
    env.backends.onnx.wasm.wasmPaths = '/vendor/';
    env.backends.onnx.wasm.numThreads = variant.threads;
    env.backends.onnx.wasm.proxy = false;
    if (adapter) env.backends.onnx.webgpu.adapter = adapter;
    const kernels = {};
    if (variant.profile) {
      // Also verify real GPU dispatches independently of the runtime's profiling API.
      // This works with both the native WebGPU and older JSEP runtime builds.
      const dispatches = {};
      const pipelines = new WeakMap(), passes = new WeakMap();
      const shaders = new WeakMap();
      const createShader = GPUDevice.prototype.createShaderModule;
      GPUDevice.prototype.createShaderModule = function (descriptor) {
        const shader = createShader.call(this, descriptor);
        shaders.set(shader, descriptor.label || descriptor.code.match(/fn\s+(\w+)/)?.[1] || 'unlabelled');
        return shader;
      };
      for (const method of ['createComputePipeline', 'createComputePipelineAsync']) {
        const original = GPUDevice.prototype[method];
        GPUDevice.prototype[method] = function (descriptor) {
          const result = original.call(this, descriptor);
          const label = descriptor.label || shaders.get(descriptor.compute.module) || descriptor.compute.entryPoint || 'unlabelled';
          if (result instanceof Promise) return result.then(pipeline => { pipelines.set(pipeline, label); return pipeline; });
          pipelines.set(result, label); return result;
        };
      }
      const setPipeline = GPUComputePassEncoder.prototype.setPipeline;
      GPUComputePassEncoder.prototype.setPipeline = function (pipeline) {
        passes.set(this, pipelines.get(pipeline) || 'unlabelled');
        return setPipeline.call(this, pipeline);
      };
      for (const method of ['dispatchWorkgroups', 'dispatchWorkgroupsIndirect']) {
        const original = GPUComputePassEncoder.prototype[method];
        GPUComputePassEncoder.prototype[method] = function (...args) {
          const label = passes.get(this) || 'unlabelled';
          dispatches[label] = (dispatches[label] || 0) + 1;
          return original.apply(this, args);
        };
      }
      result.gpuDispatches = dispatches;
      env.backends.onnx.webgpu.profiling = { mode: 'default', ondata: event => {
        const key = event.kernelType ?? event.kernelName ?? event.programName ?? 'unknown';
        kernels[key] = (kernels[key] ?? 0) + 1;
      } };
    }
    progress('loading encoder');
    const loadStart = performance.now();
    pipe = await pipeline('feature-extraction', 'encoder', {
      device: variant.device, dtype: variant.dtype, local_files_only: true,
      session_options: { executionProviders: [variant.device] },
    });
    result.loadMs = performance.now() - loadStart;
    const indices = Uint32Array.from({ length: manifest.features.dimensions }, (_, i) => i);
    const predict = async texts => {
      // Match the production extractor: length check, mean pooling, normalization,
      // float32 vectors, the shared head, temperature, and acceptance policy.
      for (const text of texts) {
        const tokens = pipe.tokenizer(text, { truncation: false, padding: false, return_tensor: false });
        if (tokens.input_ids.length > manifest.features.maxTokens) throw new Error('Input exceeds model token limit');
      }
      const tensor = await pipe(texts, { pooling: 'mean', normalize: true });
      return tensor.tolist().map(values => {
        if (values.length !== indices.length || values.some(value => !Number.isFinite(value))) throw new Error('Invalid embedding');
        const probabilities = predictHead(manifest.head, { indices, values: Float32Array.from(values) }, manifest.calibration.temperature);
        const winner = probabilities.indexOf(Math.max(...probabilities));
        // Explicit benchmark policy; the inference SDK returns probabilities only.
        const cutoff = manifest.thresholdRecommendation.threshold;
        const accepted = cutoff !== null && probabilities[winner] >= cutoff;
        return { choice: manifest.task.labels[winner], probabilities, accepted };
      });
    };
    const rows = reference.rows;
    progress('first prediction');
    const first = performance.now();
    await predict([rows[0].text]);
    result.firstPredictionMs = performance.now() - first;
    if (variant.profile) {
      // Profiling is a separate run; its overhead is never included in latency results.
      await predict(rows.slice(0, 32).map(row => row.text));
      await pipe.dispose(); pipe = undefined;
      result.gpuKernelEvents = kernels;
    } else {
      progress('warming up and timing 200 single predictions');
      for (let i = 0; i < 20; i++) await predict([rows[i].text]);
      const times = [];
      for (let i = 0; i < 200; i++) {
        const start = performance.now();
        await predict([rows[Math.floor(i * rows.length / 200)].text]);
        times.push(performance.now() - start);
      }
      result.single = stats(times);
      progress('timing 30 batches of 32 inputs');
      const batches = [];
      for (let i = 0; i < 35; i++) {
        const offset = i * 32 % (rows.length - 32);
        const texts = rows.slice(offset, offset + 32).map(row => row.text);
        const start = performance.now();
        await predict(texts);
        if (i >= 5) batches.push(performance.now() - start);
      }
      result.batch32 = { ...stats(batches), inputsPerSecond: 32_000 / stats(batches).mean };
      if (variant.evaluateAccuracy !== false) {
        let correct = 0, accepted = 0, acceptedCorrect = 0, sameChoice = 0, sameAcceptance = 0;
        let probabilityAbsSum = 0, probabilityMaxDifference = 0;
        for (let start = 0; start < rows.length; start += 32) {
          if (start % 320 === 0) progress(`accuracy and parity: ${start}/${rows.length}`);
          const slice = rows.slice(start, start + 32);
          const predictions = await predict(slice.map(row => row.text));
          predictions.forEach((p, i) => {
            const row = slice[i];
            correct += +(p.choice === row.label); accepted += +p.accepted; acceptedCorrect += +(p.accepted && p.choice === row.label);
            sameChoice += +(p.choice === row.reference.choice); sameAcceptance += +(p.accepted === row.reference.accepted);
            p.probabilities.forEach((value, j) => {
              const delta = Math.abs(value - row.reference.probabilities[manifest.task.labels[j]]);
              probabilityAbsSum += delta; probabilityMaxDifference = Math.max(probabilityMaxDifference, delta);
            });
          });
        }
        result.accuracy = { examples: rows.length, correct, accepted, acceptedCorrect, accuracy: correct / rows.length,
          coverage: accepted / rows.length, acceptedAccuracy: accepted ? acceptedCorrect / accepted : null };
        result.nodeQ8Parity = { sameChoice, sameAcceptance, choiceAgreement: sameChoice / rows.length,
          acceptanceAgreement: sameAcceptance / rows.length, meanAbsoluteProbabilityDifference: probabilityAbsSum / (rows.length * manifest.task.labels.length),
          maxAbsoluteProbabilityDifference: probabilityMaxDifference };
      }
    }
  } catch (error) { result.error = String(error.message ?? error); result.stack = error.stack; }
  finally { if (pipe) await pipe.dispose().catch(() => {}); }
  result.resources = performance.getEntriesByType('resource').map(entry => ({
    path: new URL(entry.name).pathname, transferSize: entry.transferSize, decodedBodySize: entry.decodedBodySize,
  }));
  postMessage(result);
};

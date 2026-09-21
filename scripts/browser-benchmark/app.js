const status = document.querySelector('#status');
const button = document.querySelector('#run');
const results = document.querySelector('#results');
const details = document.querySelector('#details');
const config = await (await fetch('/config.json')).json();
const variants = [
  { name: 'q8 / WASM 1 thread', dtype: 'q8', device: 'wasm', threads: 1 },
  { name: 'q8 / WebGPU', dtype: 'q8', device: 'webgpu', threads: 1 },
  { name: 'fp32 / WASM 1 thread', dtype: 'fp32', device: 'wasm', threads: 1 },
  { name: 'fp32 / WebGPU', dtype: 'fp32', device: 'webgpu', threads: 1 },
  { name: 'fp16 / WebGPU', dtype: 'fp16', device: 'webgpu', threads: 1 },
  { name: 'q8 / WASM 4 threads', dtype: 'q8', device: 'wasm', threads: 4 },
];
status.textContent = 'Ready. All inference stays on this computer.';
button.disabled = false;
button.onclick = async () => {
  button.disabled = true;
  results.replaceChildren();
  const suite = document.querySelector('#suite').value;
  const profile = suite === 'profile';
  const report = { createdAt: new Date().toISOString(), suite,
    userAgent: navigator.userAgent, hardwareConcurrency: navigator.hardwareConcurrency,
    crossOriginIsolated, variants: [], visibilityChanged: false };
  const visibility = () => { report.visibilityChanged = true; };
  document.addEventListener('visibilitychange', visibility);
  try {
    const selected = variants.filter(v => profile ? v.device === 'webgpu' : suite === 'latency'
      ? (v.dtype === 'q8' && v.device === 'wasm' && v.threads === 1) || v.dtype === 'fp16' : true);
    for (const variant of selected) {
      status.textContent = `${variant.name}: starting worker…`;
      const row = document.createElement('tr'); results.append(row);
      const worker = new Worker('/worker.js', { type: 'module' });
      const began = performance.now();
      const result = await new Promise(resolve => {
        const timeout = setTimeout(() => resolve({ ...variant, error: 'Variant exceeded 10 minutes' }), 600_000);
        worker.onmessage = ({ data }) => {
          if (data.progress) { status.textContent = `${variant.name}: ${data.progress}`; return; }
          clearTimeout(timeout); resolve(data);
        };
        worker.onerror = event => { clearTimeout(timeout); resolve({ ...variant, error: event.message }); };
        worker.postMessage({ ...variant, profile, evaluateAccuracy: suite === 'timing' });
      });
      worker.terminate();
      result.workerWallMs = performance.now() - began;
      report.variants.push(result);
      const ms = value => value === undefined ? '—' : `${value.toFixed(2)} ms`;
      const cells = [variant.name, ms(result.loadMs), ms(result.firstPredictionMs),
        result.single ? `${ms(result.single.p50)} / ${ms(result.single.p95)}` : result.error ?? 'Profiled',
        result.batch32?.inputsPerSecond?.toFixed(1) ?? '—',
        result.accuracy ? `${(100 * result.accuracy.accuracy).toFixed(2)}%` : '—'];
      for (const value of cells) { const cell = document.createElement('td'); cell.textContent = value; row.append(cell); }
      details.textContent = JSON.stringify(report, null, 2);
    }
    const response = await fetch('/results', { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-Benchmark-Token': config.token }, body: JSON.stringify(report) });
    if (!response.ok) throw new Error(`Saving failed: HTTP ${response.status}`);
    status.textContent = `Complete. Saved to ${(await response.json()).saved}`;
  } catch (error) { status.textContent = `Error: ${error.message}`; }
  finally { document.removeEventListener('visibilitychange', visibility); button.disabled = false; }
};

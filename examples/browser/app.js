import { loadClassifier } from '/jimothy/browser.js';
import { parseInputs, makePrediction } from '/inference.js';
const $ = id => document.getElementById(id);
const pretty = label => { const text = label.replaceAll('_', ' ').replace(/\?$/, ''); return text.charAt(0).toUpperCase() + text.slice(1); };
const percent = n => `${(n * 100).toFixed(1)}%`;
let classifier, loading, config, loaded, busy = false, inputMode = 'single';

function updateButton() {
  $('classify').disabled = !loaded || busy || !$('message').value.trim();
  $('classify').textContent = busy ? 'Classifying…' : 'Classify';
}

function clearResults() {
  $('result-panel').hidden = true;
  $('error').hidden = true;
}

function inputChanged() { clearResults(); updateButton(); }
function showError(message) { $('error').textContent = message; $('error').hidden = false; }

function switchInput(mode) {
  inputMode = mode;
  for (const value of ['single', 'batch']) $(`${value}-mode`).setAttribute('aria-pressed', String(value === mode));
  $('message').setAttribute('aria-label', mode === 'batch' ? 'Messages, one per line' : 'Message');
  $('message').placeholder = mode === 'batch' ? 'One message per line' : 'Message';
  inputChanged();
}

function render(data) {
  const first = data.predictions[0];
  $('single-result').hidden = inputMode !== 'single';
  $('batch-result').hidden = inputMode !== 'batch';
  if (inputMode === 'single') {
    $('rankings').replaceChildren();
    for (const item of first.ranked) {
      const li = document.createElement('li'), name = document.createElement('span'), score = document.createElement('span');
      name.textContent = pretty(item.label);
      score.textContent = percent(item.probability);
      li.append(name, score);
      $('rankings').append(li);
    }
  } else {
    $('batch-rows').replaceChildren();
    data.predictions.forEach((prediction, i) => {
      const row = document.createElement('tr'), text = document.createElement('td'), confidence = document.createElement('td');
      const message = document.createElement('span'), label = document.createElement('strong');
      message.className = 'message-text';
      message.textContent = data.texts[i];
      label.textContent = pretty(prediction.choice);
      text.append(message, label);
      confidence.textContent = percent(prediction.maxProbability);
      row.append(text, confidence);
      $('batch-rows').append(row);
    });
  }
  $('latency').textContent = `${data.inferenceMs.toFixed(1)} ms${data.predictions.length > 1 ? ' total' : ''}`;
  $('latency').title = 'Inference time, including tokenization. Model loading excluded.';
  $('json-output').textContent = JSON.stringify(data.predictions.length === 1 ? first.response : data.predictions.map(p => p.response), null, 2);
  $('result-panel').hidden = false;
}

async function load() {
  if (!config) return;
  loading?.abort(); classifier?.dispose(); classifier = undefined;
  const abort = new AbortController(); loading = abort;
  loaded = null; busy = false;
  clearResults(); updateButton();
  $('result-panel').setAttribute('aria-busy', 'false');
  $('reload').hidden = true; $('status').textContent = 'Loading…';
  $('status').removeAttribute('title');
  try {
    const model = await loadClassifier(config.base, { assetsUrl: config.assetsUrl, device: $('engine').value,
      webgpu: config.webgpu, signal: abort.signal, onProgress: () => { $('status').textContent = 'Loading…'; } });
    if (loading !== abort) { await model.dispose(); return; }
    classifier = model; loaded = model.metadata;
    $('status').textContent = loaded.device === 'webgpu' ? 'WebGPU · FP16' : 'WASM · q8';
    $('status').title = [`Loaded in ${(loaded.loadMs / 1000).toFixed(2)} s`, loaded.fallbackReason].filter(Boolean).join('. ');
  } catch (error) {
    if (loading !== abort) return;
    $('status').textContent = 'Model unavailable'; $('reload').hidden = false;
    showError(error.message);
  }
  updateButton();
}

$('classify-form').addEventListener('submit', async event => {
  event.preventDefault();
  if (!loaded || busy || !$('message').value.trim()) return;
  clearResults(); busy = true; updateButton();
  $('result-panel').setAttribute('aria-busy', 'true');
  const model = classifier, text = $('message').value, mode = inputMode;
  try {
    const texts = parseInputs(text, mode), started = performance.now();
    const results = await model.predictBatch(texts);
    if (classifier === model && text === $('message').value && mode === inputMode) {
      render({ texts, predictions: results.map(prediction => makePrediction(prediction, model.metadata)), inferenceMs: performance.now() - started });
      $('reload').hidden = true;
    }
  } catch (error) { if (classifier === model) showError(error.message); }
  finally { if (classifier === model) { busy = false; $('result-panel').setAttribute('aria-busy', 'false'); updateButton(); } }
});
$('message').addEventListener('input', inputChanged);
$('message').addEventListener('keydown', event => {
  if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') {
    event.preventDefault(); $('classify-form').requestSubmit();
  }
});
$('single-mode').onclick = () => switchInput('single');
$('batch-mode').onclick = () => switchInput('batch');
for (const button of document.querySelectorAll('[data-example]')) button.onclick = () => {
  $('message').value = button.dataset.example;
  switchInput('single');
  $('message').focus();
};
$('engine').onchange = () => load();
$('reload').onclick = () => load();
window.addEventListener('pagehide', () => { loading?.abort(); classifier?.dispose(); });
inputChanged();
if (location.protocol === 'file:') {
  showError('Run npm run demo:browser and open http://127.0.0.1:4319.');
} else {
  try {
    const response = await fetch('/config.json');
    if (!response.ok) throw new Error('Could not load configuration. Reload the page to try again.');
    config = await response.json();
    document.title = `${config.taskName} · Local classifier`;
    if (!config.webgpu) $('engine').querySelector('[value="webgpu"]').disabled = true;
    if (config.taskName === 'Custom classifier') { $('message').value = ''; inputChanged(); }
    else $('examples').hidden = false;
    load();
  } catch (error) {
    showError(error.message);
    $('status').textContent = 'Server unavailable';
  }
}

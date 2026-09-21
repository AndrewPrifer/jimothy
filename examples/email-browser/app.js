import { loadClassifier } from '/jimothy/browser.js';
import { emailState, examples } from '/email.js';

const $ = id => document.getElementById(id);
let config, classifier, loading, loaded = false, busy = false, revision = 0, controller;
let responses = {};
let expiryTimer;
const teacherReady = () => Boolean(config?.teacherReady || $('api-key').value.trim());
function expireSharedKey() {
  config.teacherReady = false;
  $('api-key-field').hidden = false;
  buttons();
}
function scheduleExpiry() {
  $('api-key-field').hidden = config.teacherReady;
  if (!config.teacherReady) return;
  const remaining = config.sharedKeyExpiresAt - config.serverTime;
  const deadline = performance.now() + remaining;
  const tick = () => {
    const delay = deadline - performance.now();
    if (delay <= 0) expireSharedKey();
    else expiryTimer = setTimeout(tick, Math.min(delay, 2_147_483_647));
  };
  tick();
}
const pretty = label => label.charAt(0).toUpperCase() + label.slice(1);

function buttons() {
  const hasBody = Boolean($('body').value.trim());
  $('local').disabled = busy || !loaded || !hasBody;
  $('jev').disabled = busy || !teacherReady() || !hasBody;
  $('compare').disabled = busy || !loaded || !teacherReady() || !hasBody;
}
function clear() {
  revision++;
  controller?.abort();
  $('results').hidden = true; $('json-panel').hidden = true; $('error').hidden = true;
  responses = {};
  for (const button of $('examples').children) {
    const sample = examples.find(example => example.name === button.textContent);
    button.setAttribute('aria-pressed', String(['from', 'subject', 'body'].every(key => $(key).value === sample[key])));
  }
  buttons();
}
function error(message) { $('error').textContent = message; $('error').hidden = false; }
function fill(sample) {
  for (const key of ['from', 'subject', 'body']) $(key).value = sample[key];
  clear();
}
function panel(name, message = '') {
  $(`${name}-panel`).hidden = false;
  $(`${name}-status`).textContent = message;
  $(`${name}-status`).className = '';
  $(`${name}-status`).hidden = !message;
  $(`${name}-time`).textContent = '';
  $(`${name}-ranking`).replaceChildren();
}
function render(name, response, ms) {
  panel(name);
  const answer = response.answers[config.questionId];
  const ranked = Object.entries(answer.probabilities).sort((a, b) => b[1] - a[1]);
  for (const [label, probability] of ranked) {
    const row = document.createElement('li'), text = document.createElement('span'), score = document.createElement('span');
    row.classList.toggle('winner', label === answer.choice);
    row.style.setProperty('--probability', `${Math.max(0, Math.min(100, probability * 100))}%`);
    text.textContent = pretty(label); score.textContent = `${(probability * 100).toFixed(1)}%`;
    row.append(text, score); $(`${name}-ranking`).append(row);
  }
  $(`${name}-time`).textContent = `${ms.toFixed(1)} ms`;
  $(`${name}-time`).title = name === 'local' ? 'Browser inference time. Model loading excluded.' : 'Round-trip time including the server and Jev API.';
  responses[name] = response;
  $('json').textContent = JSON.stringify(responses, null, 2);
  $('json-panel').hidden = false;
}
function failed(name, message) {
  panel(name, message);
  $(`${name}-status`).className = 'failed';
}
async function load() {
  loading?.abort(); classifier?.dispose(); classifier = undefined;
  const abort = new AbortController(); loading = abort;
  loaded = false; clear(); $('reload').hidden = true;
  $('status').textContent = 'Loading model…';
  try {
    const model = await loadClassifier(config.base, { assetsUrl: config.assetsUrl, signal: abort.signal,
      onProgress: () => { $('status').textContent = 'Loading model…'; } });
    if (loading !== abort) { await model.dispose(); return; }
    classifier = model; loaded = true;
    $('status').textContent = '';
  } catch (cause) {
    if (loading !== abort) return;
    $('status').textContent = 'Local model unavailable'; $('reload').hidden = false;
    error(cause.message);
  }
  buttons();
}

async function classify(mode) {
  if (busy || !config || (mode !== 'jev' && !loaded) || (mode !== 'local' && !teacherReady())) return;
  clear();
  let state;
  try { state = emailState({ from: $('from').value, subject: $('subject').value, body: $('body').value }); }
  catch (cause) { error(cause.message); return; }
  const version = revision;
  busy = true; buttons(); responses = {};
  $('results').hidden = false; $('results').dataset.mode = mode;
  $('local-panel').hidden = mode === 'jev'; $('jev-panel').hidden = mode === 'local';
  controller = new AbortController();
  const jobs = [];
  if (mode !== 'jev') {
    panel('local', 'Classifying…');
    jobs.push((async () => {
      try {
        const started = performance.now();
        const response = await classifier.evaluate({ state });
        if (revision === version) render('local', response, performance.now() - started);
      } catch (cause) { if (revision === version) failed('local', cause.message); }
    })());
  }
  if (mode !== 'local') {
    panel('jev', 'Classifying…');
    jobs.push((async () => {
      const started = performance.now();
      try {
        const response = await fetch('/api/jev', { method: 'POST', headers: { 'Content-Type': 'application/json', ...($('api-key').value.trim() ? { 'X-Jev-Api-Key': $('api-key').value.trim() } : {}) },
          body: JSON.stringify(state), signal: controller.signal });
        const result = await response.json();
        if (result.code === 'api_key_required') expireSharedKey();
        if (!response.ok) throw new Error(result.error ?? 'Jev is unavailable. Try again.');
        if (revision === version) render('jev', result, performance.now() - started);
      } catch (cause) { if (revision === version) failed('jev', cause.name === 'AbortError' ? 'Request cancelled.' : cause.message); }
    })());
  }
  await Promise.allSettled(jobs);
  busy = false; controller = null; buttons();
}

for (const key of ['from', 'subject', 'body']) $(key).addEventListener('input', clear);
for (const sample of examples) {
  const button = document.createElement('button'); button.type = 'button'; button.textContent = sample.name;
  button.onclick = () => fill(sample); $('examples').append(button);
}
$('email-form').onsubmit = event => { event.preventDefault(); classify(event.submitter?.value ?? 'both'); };
$('body').onkeydown = event => {
  if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') {
    event.preventDefault(); if (!$('compare').disabled) $('email-form').requestSubmit($('compare'));
  }
};
$('reload').onclick = load;
$('api-key').addEventListener('input', clear);
window.addEventListener('pagehide', () => { clearTimeout(expiryTimer); $('api-key').value = ''; controller?.abort(); loading?.abort(); classifier?.dispose(); });
fill(examples.at(-1));
if (location.protocol !== 'file:') {
  try {
    const response = await fetch('/config.json');
    if (!response.ok) throw new Error('Could not load the email model configuration.');
    config = await response.json(); scheduleExpiry(); load();
  } catch (cause) { $('status').textContent = ''; error(cause.message); }
}

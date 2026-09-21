import { loadClassifier } from 'jimothy/browser';

const output = document.querySelector('pre');
const fixtures = await (await fetch('/fixtures.json')).json();
let passed = 0, failed = 0;
const assert = (condition, message) => { if (!condition) throw new Error(message); };
const same = (actual, expected) => assert(JSON.stringify(actual) === JSON.stringify(expected), `Mismatch: ${JSON.stringify(actual)} / ${JSON.stringify(expected)}`);
async function rejects(promise, pattern) {
  try { await promise; } catch (error) { assert(pattern.test(error.message), error.message); return; }
  throw new Error('Expected rejection.');
}
async function check(name, run) {
  try { await run(); passed++; output.textContent += `PASS ${name}\n`; }
  catch (error) { failed++; output.textContent += `FAIL ${name}: ${error.message}\n`; }
}

for (const fixture of fixtures) await check(`${fixture.name}: Node/browser comparison, metadata and Jev answer shape`, async () => {
  const model = await loadClassifier(`/models/${fixture.name}/`, { onProgress() { throw new Error('Observer errors are isolated'); } });
  try {
    const results = await model.predictBatch(fixture.states);
    let maxDifference = 0;
    for (let i = 0; i < results.length; i++) {
      const expected = fixture.predictions[i], actual = results[i];
      same(Object.keys(actual).sort(), ['answer', 'maxProbability']);
      same(actual.answer.type, expected.answer.type);
      if (actual.answer.type === 'choice') same(actual.answer.choice, expected.answer.choice);
      const values = a => a.probabilities ? Object.values(a.probabilities) : [a.probability ?? a.noul];
      assert(values(actual.answer).every(p => Number.isFinite(p) && p >= 0 && p <= 1), 'Invalid probability');
      if (actual.answer.probabilities) {
        assert(Math.abs(values(actual.answer).reduce((sum, p) => sum + p, 0) - 1) < 1e-10, 'Probabilities do not sum to one');
        assert(Math.abs(actual.maxProbability - Math.max(...values(actual.answer))) < 1e-10, 'Confidence does not match distribution');
      }
      values(actual.answer).forEach((p, j) => { maxDifference = Math.max(maxDifference, Math.abs(p - values(expected.answer)[j])); });
      assert(Number.isFinite(actual.maxProbability) && actual.maxProbability >= 0 && actual.maxProbability <= 1, 'Invalid confidence');
      if (actual.answer.type === 'score') assert(Math.abs(actual.answer.score - expected.answer.score) < 1e-4, 'Score mismatch');
    }
    // Native ONNX and WASM q8 are known to differ numerically. Record drift for MiniLM;
    // require exact shared-core parity for TF-IDF, and matching sample labels for both.
    if (fixture.name !== 'email') assert(maxDifference === 0, `Probability delta ${maxDifference}`);
    const evaluated = await model.evaluate({ state: fixture.states[0] });
    same(Object.keys(evaluated).sort(), ['answers', 'model']);
    same(evaluated.model, fixture.metadata.id);
    same(model.metadata.thresholdRecommendation, fixture.metadata.thresholdRecommendation);
    same(await model.predictBatch([]), []);
    const copy = model.metadata; copy.task.labels.length = 0;
    assert(model.metadata.task.labels.length > 0, 'Metadata mutated internal model');
    const concurrent = await Promise.all(fixture.states.map(state => model.predict(state)));
    assert(concurrent.length === fixture.states.length, 'Concurrent request lost');
    await rejects(model.predict('good '.repeat(fixture.name === 'email' ? 300 : 5000)), /supports/);
    await model.predict(fixture.states[0]); // An input error must not poison later requests.
    output.textContent += `  max probability delta: ${maxDifference.toExponential(2)}; device: ${model.metadata.device}\n`;
  } finally { await model.dispose(); }
  await rejects(model.predict('good'), /disposed/);
});
await check('independent workers and canonical structured inputs', async () => {
  const [a, b] = await Promise.all([loadClassifier('/models/choice/'), loadClassifier('/models/boolean/')]);
  try {
    same(await a.predict({ b: 'bad', a: 'good' }), await a.predict({ a: 'good', b: 'bad' }));
    const pending = a.predict('good'); const rejected = rejects(pending, /disposed/);
    await a.dispose(); await rejected;
    assert((await b.predict('good')).answer.type === 'boolean', 'Disposing one model affected another');
  } finally { await a.dispose(); await b.dispose(); }
});
await check('cancellation during load', async () => {
  const abort = new AbortController();
  const pending = loadClassifier('/models/email/', { signal: abort.signal }); abort.abort();
  await rejects(pending, /abort|cancel/i);
});
await check('missing worker rejects loading', async () => {
  await rejects(loadClassifier('/models/choice/', { assetsUrl: '/missing/' }), /worker|script/i);
});
await check('corrupt report rejected before inference', () => rejects(loadClassifier('/models/damaged/'), /Checksum mismatch/));
await check('automatic fallback starts a fresh worker', async () => {
  const model = await loadClassifier('/models/choice/', { device: 'auto', webgpu: { policyUrl: '/no-policy.json', modelUrl: '/no-model.onnx' } });
  try { assert(model.metadata.device === 'javascript' && model.metadata.fallbackReason.includes('TF-IDF'), 'Missing fallback metadata'); await model.predict('good'); }
  finally { await model.dispose(); }
});
document.title = failed ? 'Browser SDK tests failed' : 'Browser SDK tests passed';
output.textContent += `\n${passed} passed, ${failed} failed. crossOriginIsolated=${crossOriginIsolated}\n`;

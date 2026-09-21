# Browser SDK

`jimothy/browser` loads an exported classifier from a URL and runs it in a dedicated module worker. It supports TF-IDF and MiniLM, all four answer types, and structured JSON states. Training stays in the CLI. Install with `npm install jimothy`.

## Setup

In an app with a `public/` directory (such as Vite or Next.js), prepare the runtime and copy your **whole** model bundle:

```sh
npx jimothy prepare-browser --out public/jimothy
mkdir -p public/models
cp -R /path/to/trained/email-model public/models/email
```

`prepare-browser` copies the SDK worker, a standalone client module, matching Transformers.js/ONNX browser assets, runtime version metadata, and license notices from the installed package. It requires optional dependencies to be installed, makes no network requests, and refuses to overwrite an existing directory. Use a new versioned directory when upgrading, then change `assetsUrl`. Keep its contents together. TF-IDF never downloads or imports the encoder runtime in the browser.

```ts
import { loadClassifier } from 'jimothy/browser';

const classifier = await loadClassifier('/models/email/');
try {
  const prediction = await classifier.predict({
    from: 'shop@example.com',
    subject: 'Your order shipped',
    body: 'Your order is on its way.',
  });
  console.log(prediction.answer, prediction.maxProbability);
  console.log(classifier.metadata.thresholdRecommendation);

  const batch = await classifier.predictBatch(['First message', 'Second message']);
  const response = await classifier.evaluate({ state: 'Your order shipped' });
  // { model, answers: { [questionId]: answer } }
} finally {
  await classifier.dispose();
}
```

Use the same state shape as training: the string examples above demonstrate the methods, while the email model was trained with `from`, `subject`, and `body`. `modelUrl` accepts a directory or a `model.json` URL. Relative URLs resolve against the page URL. Object keys are sorted recursively; array order is preserved, just as in Node.

For plain HTML, import the copied module instead:

```html
<script type="module">
  import { loadClassifier } from '/jimothy/browser.js';
  const classifier = await loadClassifier('/models/email/');
  // Use classifier.predict(...) from your UI; dispose it when finished.
  window.addEventListener('pagehide', () => classifier.dispose());
</script>
```

## Loading and lifecycle

```ts
const abort = new AbortController();
const classifier = await loadClassifier('/models/email/model.json', {
  assetsUrl: '/jimothy/',        // default; must be on the page's origin
  device: 'wasm',                // default; 'auto' and 'webgpu' are also supported
  signal: abort.signal,          // abort.abort() cancels loading
  onProgress: ({ phase, file, message }) => console.log(message),
});
```

`predict`, `predictBatch`, `evaluate`, and `metadata.thresholdRecommendation` have the same shapes as the Node SDK. The browser also exposes `metadata.device` (`javascript`, `wasm`, or `webgpu`), `dtype`, `loadMs`, and an optional `fallbackReason`. `loadMs` includes worker startup, asset downloads/checksums and runtime initialization; first inference may still need warm-up.

Calls on one classifier are queued, so concurrent predictions do not overlap ONNX sessions. Each classifier has an independent worker and runtime. `dispose()` terminates it, rejects pending predictions, and can be called more than once. The loading signal stops applying once loading completes. Dispose models when their screen unmounts. Importing the browser module during server rendering is safe; call `loadClassifier` only on the client.

There is no acceptance decision or threshold option. Recommendations are advisory. The shared head and TF-IDF calculations match Node; native ONNX and browser WASM can produce different MiniLM probabilities despite using the same q8 weights. Check held-out predictions in your deployment runtime before relying on a cutoff. The [earlier full-dataset comparison](browser-benchmark.md) documents that difference.

## WebGPU

WASM uses q8 MiniLM on one CPU thread. TF-IDF uses ordinary JavaScript in the worker. WebGPU requires a separate FP16 encoder and calibration policy tied to this exact model:

```ts
const classifier = await loadClassifier('/models/banking/', {
  device: 'auto',
  webgpu: {
    modelUrl: '/models/banking-fp16/model_fp16.onnx',
    policyUrl: '/models/banking-fp16/policy.json',
  },
});
```

`auto` tries WebGPU only when `webgpu` is provided. It uses a fresh WASM worker if GPU loading fails. Explicit `webgpu` reports failure instead. GPU adapter support must include `shader-f16`. Predictions use the FP16 policy's temperature and advisory recommendation; q8 fallback retains the base model's metadata. The FP16 model ID has a `-fp16-webgpu` suffix.

This version accepts the [existing experimental FP16 calibration export](browser-calibration.md), including `policy.json` from that workflow or the `policy` object extracted from its saved report. The loader verifies the source manifest, head, tokenizer, encoder bytes, precision, pooling, preprocessing and runtime versions. It does not generate or recalibrate an FP16 export. The base q8 bundle is also verified, so this first implementation downloads both encoder variants when loading WebGPU. HTTP caching can amortize that cost.

## Hosting

- Use HTTPS or localhost. `file://` URLs are unsupported; checksum verification needs Web Crypto.
- Serve the copied `.js`/`.mjs` files as JavaScript and `.wasm` as `application/wasm`. Do not route missing assets to your app's HTML fallback.
- Runtime assets must be on the page's origin. Model files may be elsewhere if their server allows CORS. URLs with credentials, query strings or fragments are unsupported in this version.
- No COOP/COEP headers are required for the default single-thread WASM runtime. If you set a Content Security Policy, allow your runtime scripts and workers, `wasm-unsafe-eval`, and model download origins. The examples include a working policy.
- Model bundles must use format version 3; retrain older bundles with the current CLI. Use versioned directories for models and runtime assets before enabling immutable caching. Copy the entire model bundle, including `report.json`; all manifest-listed checksums are verified. Checksums detect corruption, not authenticity.
- Model assets are fetched from the URLs you provide. Inputs stay in the worker. No provider, Hugging Face or runtime CDN requests are made. Persistent offline availability requires your app's own caching/service worker.

Verified in the desktop Chromium browser with WASM and FP16 WebGPU. Safari, Firefox and mobile still need testing; the earlier [performance results](browser-benchmark.md) are device-specific.

## Package integration test

From this repository, pack and install into a separate temporary app:

```sh
npm pack --pack-destination /path/to/consumer
cd /path/to/consumer
npm install ./jimothy-0.1.0.tgz
cd /path/to/this/repository
node scripts/browser-sdk/server.mjs /path/to/consumer
```

Open the printed localhost URL. The page tests a bundled `jimothy/browser` import, installed CLI asset preparation, all answer types, structured inputs, independent workers, concurrent calls, input limits, cancellation, disposal, damaged bundles and automatic fallback. It compares Node results with the browser, requiring exact TF-IDF parity and reporting MiniLM probability drift. The email comparison is included when `models/email-300/minilm` exists. Tests run without cross-origin isolation.

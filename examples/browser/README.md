# Browser playground

An interactive example for the trained local classifier. Enter a single message or up to 50 messages (one per line), inspect the top labels, probabilities and inference time. A collapsed panel shows the Jev-compatible result.

From the repository root:

```sh
npm run demo:browser
```

Open **http://127.0.0.1:4319**. Use the localhost URL, not `file://…/index.html`. The command builds the browser SDK and starts a static server bound to this machine. Stop it with Ctrl-C. `DEMO_PORT=4320 npm run demo:browser` selects another port.

Follow [dataset preparation](../../docs/datasets.md) to create `models/banking77-auto-v2/minilm`. That q8 model is sufficient to run the demo. The SDK requires a format-v3 bundle; retrain older models into a new directory and set `DEMO_MODEL_DIR` to that path. The [browser benchmark](../../docs/browser-benchmark.md) downloads the extra FP16 weights; GPU inference is enabled only if the model, encoder, tokenizer and runtime match the [validated FP16 policy](../../docs/browser-calibration.md).

- **Auto** tries the calibrated FP16 WebGPU model when the browser supports it. It falls back to q8 WebAssembly if GPU initialization is unavailable or fails.
- **WebGPU** explicitly requests FP16 and reports a loading error instead of silently selecting CPU.
- **WASM** runs q8 on one CPU thread.

Each backend uses its own temperature and exposes an advisory cutoff in `/config.json`: the current FP16 model uses **0.836752 / 0.90**; q8 uses its original **0.823122 / 0.85**. The demo never swaps calibration between them and does not apply an acceptance policy. The Node server validates model checksums and the FP16 policy's provenance before serving it. Versioned asset URLs permit browser HTTP caching without reusing stale model/policy pairs.

All tokenization and inference run in a dedicated browser worker. The Node process only serves static assets and accepts GET/HEAD requests. Typed messages are not sent to it or to an external service, and they are not saved. The model is a banking-intent classifier, so arbitrary non-banking text can still receive a banking label; confidence is not an out-of-domain detector.

Input longer than 256 wordpieces (including special tokens) is rejected with a visible error. Timings include tokenization, feature extraction and the head inside the worker; hover over the backend status for loading time. First predictions can take longer, especially on WebGPU. Batch size can cause small numerical differences, as documented in the calibration report.

This example uses [`jimothy/browser`](../../docs/browser-sdk.md), which also works in your own app or on a static host. For another MiniLM **choice** bundle, set `DEMO_MODEL_DIR=/absolute/path/to/model`; the demo uses q8 unless the saved FP16 policy matches that bundle. The example doesn't train or calibrate models.

Checks for the demo helpers:

```sh
node --test examples/browser/inference.test.mjs
```

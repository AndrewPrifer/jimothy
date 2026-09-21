# Browser inference experiment

This is a reproducible browser benchmark, not a released browser SDK. It uses the existing trained BANKING77 MiniLM head and local encoder files. It makes no teacher API calls and does not upload inputs.

## Measured results

Measured September 20, 2026 on **Apple M3 Max**, macOS arm64, in the Codex in-app Chromium browser (Chrome/153.0.0.0 user agent). The browser reported 16 logical processors and a hardware Apple Metal adapter with FP16 support. Transformers.js was **4.3.0**, ONNX Runtime Web **1.31.0-dev.20260914-8d85527a0**, and the Node reference used ONNX Runtime **1.30.0**. Results are specific to these builds and this machine.

| Backend / encoder | Load | First prediction | Warm p50 | Warm p95 | Batch 32, inputs/s | Test accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| WASM, q8, 1 thread | 325 ms | 27.9 ms | 7.47 ms | 16.07 ms | 52.3 | 92.40% |
| WebGPU, q8 | 376 ms | 382.9 ms | 24.84 ms | 33.87 ms | 52.8 | 92.50% |
| WASM, FP32, 1 thread | 409 ms | 23.7 ms | 7.05 ms | 15.04 ms | 52.0 | 92.34% |
| WebGPU, FP32 | 2,145 ms | 149.8 ms | 8.95 ms | 11.65 ms | 693.8 | 92.34% |
| WebGPU, FP16 | 2,334 ms | 326.6 ms | 8.70 ms | 10.22 ms | 741.8 | 92.34% |
| WASM, q8, 4 threads | 2,722 ms | 470.9 ms | 36.38 ms | 83.34 ms | 26.3 | 92.40% |

The table is the first complete run, with 200 single predictions and 30 measured batches per variant. GPU floating-point inference improved throughput by roughly 13–14× versus one-thread WASM in this pass, but did not improve median single-query latency. It did improve the measured p95. The q8 GPU route was slower for individual queries and delivered essentially no batch gain. Four-thread WASM was slower here; this does not establish that threading is slower on every browser or machine.

Two repeats of the deployment candidates in new workers in the same browser gave:

| Repeat / backend | Load | First prediction | Warm p50 | Warm p95 | Batch 32, inputs/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1: WASM, q8, 1 thread | 331 ms | 28.4 ms | 7.75 ms | 16.69 ms | 51.4 |
| 1: WebGPU, FP16 | 371 ms | 43.1 ms | 6.43 ms | 7.24 ms | 1,833.4 |
| 2: WASM, q8, 1 thread | 328 ms | 27.9 ms | 7.90 ms | 16.22 ms | 51.6 |
| 2: WebGPU, FP16 | 368 ms | 44.7 ms | 5.63 ms | 6.27 ms | 1,841.7 |

On these repeats, FP16 GPU was about 17–29% faster at median single-query latency and 36× faster at batch throughput. The improvement is consistent with retained browser/driver shader and pipeline caches; those were not cleared between runs. All variants use dynamic input shapes. Twenty warm-up queries do not necessarily exercise every shape, and the full accuracy pass exercises more shapes before later repeats. Consequently, report both initial and repeated measurements rather than treating either as a universal steady-state speed. The repeats measure timing only, using accuracy from the first full evaluation. The [raw results](../benchmarks/browser-minilm-m3-max.json) contain every timing sample, all four runs, GPU dispatch counts and asset checksums.

Separate GPU instrumentation observed actual `dispatchWorkgroups` calls. Across one single prediction and one batch, FP32 and FP16 each dispatched 96 `MatMul` kernels; q8 dispatched 24, alongside casts, dequantization and other operations. This confirms GPU work, not that every ONNX operator runs there. The native runtime did not emit events through its JSEP-style profiling callback, so the diagnostic records WebGPU API dispatches independently. Profiling was disabled during latency measurements.

Changing backend or precision did not preserve every prediction:

| Deployment | Label changes vs Node q8 | Accept/abstain changes | Coverage | Accepted accuracy |
| --- | ---: | ---: | ---: | ---: |
| WASM q8, 1 thread | 3 / 3,079 | 19 / 3,079 | 82.14% | 97.94% |
| WebGPU q8 | 13 / 3,079 | 30 / 3,079 | 82.30% | 97.95% |
| WebGPU FP16 | 33 / 3,079 | 70 / 3,079 | 82.56% | 97.68% |

These comparisons include a runtime version change as well as execution-provider differences. Similar aggregate accuracy is not exact parity or proof that an existing calibration remains valid. The largest per-class probability difference from Node q8 was 0.173 for WASM q8 and 0.260 for WebGPU FP16. Train/calibrate using the intended encoder precision, and check deployment-runtime predictions on held-out data before shipping.

Encoder weights alone are **22.97 MB q8**, **45.30 MB FP16**, or **90.39 MB FP32**. The runtime actually fetched by these builds includes a **26.86 MB WASM binary**, even for GPU execution. With the runtime JS, tokenizer and trained head, the observed inference assets total approximately **51.8 MB for q8** or **74.1 MB for FP16**, before HTTP compression. This excludes the benchmark's test/reference fixture, unused runtime variants, training reports and app shell. It is not peak memory usage.

For the first browser SDK, use **one-thread q8 WASM as a compact, broadly compatible default**, with a worker to keep the UI responsive. Offer **FP16 WebGPU as an acceleration option**, especially for sustained batches; a warmed GPU can also improve individual-query latency. A [follow-up calibration experiment](browser-calibration.md) has now validated a separate FP16 policy on this configuration: 98.66% accepted accuracy at 77.75% coverage. Choosing an automatic default needs to account for workload, loading, download size and warm-up, rather than hardware availability alone. Test Safari, Firefox and mobile before making a cross-browser performance promise.

## Reproduce

First prepare BANKING77 and train the models using the commands in [datasets.md](datasets.md). The experiment expects `models/banking77-auto-v2/minilm` and `datasets/banking77/test.jsonl`.

```sh
npm ci
npm run build
# Download FP16 and FP32 exports at the trained encoder's immutable revision,
# and generate native Node q8 predictions for all 3,079 test examples.
node scripts/benchmark-browser.mjs prepare --download
node scripts/benchmark-browser.mjs serve
```

Open the printed loopback URL and click **Run benchmark**. Keep the tab visible. Each suite saves a new JSON file under `models/browser-benchmark/`; previous runs are preserved. Select **GPU kernel verification (untimed)** for a separate diagnostic pass, or **Repeat q8 WASM / FP16 WebGPU timing** to repeat the two most useful deployment candidates without re-evaluating accuracy. Stop the server with Ctrl-C when finished.

After the extra exports are downloaded, `prepare` without `--download` regenerates the reference entirely offline. The server rejects a stale reference when model or test-file hashes change. Browser assets are served locally from an explicit allowlist; the server binds only to `127.0.0.1`.

## Protocol

- Six sequential variants: q8 WASM with one thread, q8 WebGPU, FP32 WASM with one thread, FP32 WebGPU, FP16 WebGPU, and q8 WASM with four threads. Each receives a new dedicated worker, runtime and model session; completed workers are terminated.
- Record runtime import separately from model loading. Measure the first prediction, then 20 warm-up predictions followed by 200 sequential single predictions sampled evenly across the test set. Report p50, p95, mean and all timing samples.
- Batch size 32: five warm-up batches, then 30 timed batches containing different inputs. Throughput is total inputs divided by total measured time, including tokenization, encoder, pooling, normalization and the head. Token lengths vary, so batches pad to their longest input, as in the Node SDK.
- Timing occurs inside the worker. It includes the production token-length guard, float32 embeddings, shared `predictHead`, temperature scaling and acceptance decision. It excludes per-request main-thread messaging and Jev-compatible response-object construction. Inputs here are strings; structured JSON preprocessing is not exercised.
- Evaluate every variant on all 3,079 test examples in batches of 32. Compare labels, acceptance decisions and all class probabilities with the native Node q8 reference. No retraining, recalibration or threshold changes occur between variants. A precision change is an experiment, not an automatically validated replacement encoder.
- WebGPU is explicitly requested. GPU kernel verification is a separate run, so diagnostic overhead does not contaminate reported latency. Availability, successful execution and speed are separate findings; errors are recorded rather than silently replaced by a WASM result.
- Browser model and WASM caches are disabled, and HTTP uses `no-store`. Load times include loopback transfer and session creation, with potentially warm OS and browser shader caches. They are **not internet download estimates or guaranteed fresh-browser cold starts**.
- Four-thread WASM requires cross-origin isolation, which the server enables with COOP/COEP headers. One-thread WASM does not require this for threading. The benchmark records browser, hardware, installed runtime versions, model revision, asset sizes and SHA-256 checksums.

The test set has been inspected in earlier experiments. These results describe this task and machine, not performance or calibrated acceptance guarantees on a new dataset. Memory use, mobile browsers and end-user network download time are not measured.

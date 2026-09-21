# Performance

Reproduction commands run from the repository root.

For the [email classifier (300 samples)](email-classifier.md), local MiniLM training took **3.80 s**, with **6.80 ms p50 / 8.33 ms p95** warm single-email inference and a **23.77 MB** bundle. Agreement with Jev on 60 test emails was **75.0% in batches and 78.3% for single inputs**. Five labels differed by execution mode, and the test set had no Jev-labeled social examples. The small calibration split provides insufficient evidence to recommend a threshold; see the experiment guide for these limitations and labeling costs.

For a workload like the BANKING77 benchmark below on similar hardware, allow roughly **40 seconds for TF-IDF or 7–8 minutes for MiniLM training**, including automatic tuning, calibration, and evaluation. Once loaded, predictions take **well under a millisecond for TF-IDF and a few milliseconds for MiniLM**. These are planning estimates from one measured run, not fixed runtime guarantees.

Measured on **Apple M3 Max, macOS arm64, Node.js 25.6.0**, using local CPU inference and short English BANKING77 queries. The task has **77 classes**, with **8,023 training, 1,969 development, and 3,079 test examples**. The run searched five regularization strengths with at most **100 epochs per candidate**, then calibrated probabilities and selected acceptance thresholds targeting 95% accuracy against the supplied labels.

| Metric | TF-IDF + linear head | Frozen MiniLM + linear head |
| --- | ---: | ---: |
| Train, tune, calibrate, and evaluate | 37.6 s | 7 min 26 s |
| Warm inference, median (p50) | 0.023 ms | 1.72 ms |
| Warm inference, p95 | 0.044 ms | 3.81 ms |
| Load an existing bundle in a warm process | 30.5 ms | 59.0 ms |
| Model bundle size | 6.64 MB | 24.58 MB |
| Overall test accuracy | 82.20% | 92.37% |
| Accepted fraction (coverage) | 22.80% | 82.23% |
| Accuracy on accepted inputs | 99.57% | 97.95% |
| Macro F1, equally weighting all 77 classes | 82.13% | 92.36% |

These measurements predate the advisory API. Coverage is the share of inputs above the selected cutoff; enforcing that cutoff is now an application decision. Read it alongside accepted accuracy: TF-IDF's higher accepted accuracy here comes with much lower coverage. These results use the existing human reference labels and a previously inspected benchmark test set; they do not establish equivalent performance on a new task or teacher-generated data.

Timing and size details:

- Training time includes local feature extraction, all five head fits, calibration, threshold selection, and test evaluation. It excludes dependency installation, initial encoder downloads, teacher labelling, and final bundle serialization. **The CLI defaults to 200 epochs per candidate**, so an ordinary training run can take longer than this 100-epoch benchmark; early stopping may shorten it. More examples, classes, input tokens, or training epochs also affect runtime.
- Inference timings cover 200 sequential single-input SDK predictions after 20 warm-up calls. They include tokenization, feature extraction, the head, calibration, and the acceptance decision. They exclude process startup and model loading; p95 is the time within which 95% of those measured calls completed. They are not batched throughput or concurrent-service measurements.
- Bundle loading was measured immediately after training in the same process, with libraries and filesystem caches already warm. Fresh-process cold-start time and peak RAM have not been measured. Reuse a loaded classifier across requests.
- Bundle sizes use decimal MB and include model assets and reports. They exclude installed Node.js/ONNX runtime dependencies and process memory. These measurements cover the current linear heads; additional hidden layers have not yet been benchmarked.

See the [recorded results](../benchmarks/banking77-auto-v2.json) for exact values and settings, or the [benchmark guide](datasets.md) to reproduce the run. Reloaded SDK predictions were checked against the training report on all 3,079 test examples.

## Browser benchmark

A standalone [browser benchmark](browser-benchmark.md) compares WebAssembly and WebGPU using the same trained MiniLM head. On the M3 Max in Chromium 153, the first complete run measured:

| Browser backend | Warm p50 / p95 | Batch 32, inputs/s | Encoder weights | Test accuracy |
| --- | ---: | ---: | ---: | ---: |
| q8 WASM, 1 thread | 7.47 / 16.07 ms | 52.3 | 22.97 MB | 92.40% |
| q8 WebGPU | 24.84 / 33.87 ms | 52.8 | 22.97 MB | 92.50% |
| FP32 WASM, 1 thread | 7.05 / 15.04 ms | 52.0 | 90.39 MB | 92.34% |
| FP32 WebGPU | 8.95 / 11.65 ms | 693.8 | 90.39 MB | 92.34% |
| FP16 WebGPU | 8.70 / 10.22 ms | 741.8 | 45.30 MB | 92.34% |
| q8 WASM, 4 threads | 36.38 / 83.34 ms | 26.3 | 22.97 MB | 92.40% |

Two repeats in new workers in the same browser improved FP16 WebGPU to **5.63–6.43 ms p50, 6.27–7.24 ms p95 and 1,833–1,842 inputs/s**. q8 WASM stayed at **7.75–7.90 ms p50, 16.22–16.69 ms p95 and 51.4–51.6 inputs/s**. GPU model loading fell from 2.33 s to 0.37 s, consistent with retained browser/driver caches. Model and HTTP caching were disabled, but shader caches were not cleared; the first table is not a guaranteed cold start either. See the [raw measurements](../benchmarks/browser-minilm-m3-max.json).

**q8 WASM is the smaller initial default; FP16 WebGPU is a useful acceleration option, especially for batches.** A warmed GPU can also win for individual queries. GPU execution was verified separately. Runtime assets add download size: the observed uncompressed inference assets total about 51.8 MB for q8 and 74.1 MB for FP16, including the runtime, tokenizer and head. Loading and first-use timings, parity details, limitations and reproduction steps are in the [browser benchmark guide](browser-benchmark.md).

FP16 [calibration has now been validated](browser-calibration.md) on the reserved BANKING77 development splits and then evaluated on all 3,079 test inputs. The selected temperature is **0.836752** and the acceptance threshold is **0.90**, giving **98.66% accepted accuracy at 77.75% coverage** in batches. Single-input results are similar: **98.62% at 77.82% coverage**, with a few decisions differing across batch shapes. The higher threshold improves accepted accuracy by abstaining more; held-out probability loss was slightly worse than reusing the old temperature. The saved FP16 policy is an experimental artifact for browser integration, separate from the unchanged q8 model.

Run `node scripts/benchmark-browser.mjs prepare --download`, then `node scripts/benchmark-browser.mjs serve`, after preparing the BANKING77 model. The reusable browser API is available at `jimothy/browser`; see the [browser SDK guide](browser-sdk.md).

## Reproduce the larger benchmark

The [BANKING77 preparation and benchmark guide](datasets.md) converts 13,083 public, human-labelled banking queries into a 77-class task, training/evaluation splits, and separate Jev-compatible input/output files. It includes input-only files for a later permitted-teacher experiment.

```sh
python3 scripts/prepare-banking77.py
node scripts/benchmark-banking77.mjs
```

Build the project and prepare `.cache/minilm` first using the [benchmark guide](datasets.md#run-the-benchmark). The benchmark trains both local backends, measures held-out accuracy and SDK latency, and saves models and reports under `models/banking77-auto-v2/`.

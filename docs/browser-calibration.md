# FP16 WebGPU calibration validation

The existing BANKING77 head was recalibrated using its actual FP16 WebGPU logits. The deployment policy met the existing **95% accepted-accuracy target** using the reserved development evidence. Evaluation on all 3,079 test inputs then measured **98.66% accepted accuracy at 77.75% coverage** in batches, and **98.62% at 77.82% coverage** for individual inputs.

This is validation on the M3 Max / Chromium configuration from the [browser benchmark](browser-benchmark.md), using Transformers.js 4.3.0 and ONNX Runtime Web 1.31.0-dev.20260914-8d85527a0. BANKING77's test set had already been inspected in earlier experiments. These are follow-up results on that set, not a new blind evaluation or a guarantee for other tasks and devices.

The acceptance decisions in this experiment are an explicit benchmark policy. The SDK and example sites now return predictions without enforcing a cutoff; the calibrated cutoff is advisory metadata.

## Result

Both columns below use the same FP16 encoder and unchanged trained head. The baseline applies the original q8 model's temperature and threshold to FP16; the new policy is fitted for FP16.

| Parameter or batch-32 test metric | Original policy on FP16 | FP16 policy |
| --- | ---: | ---: |
| Temperature | 0.823122 | 0.836752 |
| Acceptance threshold | 0.85 | 0.90 |
| Overall accuracy | 92.34% | 92.34% |
| Accepted inputs | 2,542 / 3,079 | 2,394 / 3,079 |
| Coverage | 82.56% | 77.75% |
| Accepted accuracy | 97.68% | 98.66% |
| Accepted mistakes | 59 | 32 |
| Test log loss, lower is better | 0.273554 | 0.273790 |
| Multiclass Brier score, lower is better | 0.115683 | 0.115822 |
| ECE, 10 equal-width confidence bins | 0.83% | 1.11% |

The more selective threshold removes 148 accepted inputs, including 27 mistakes. Temperature scaling preserves class rankings, so recalibration itself changes no winning labels. Test probability metrics became slightly worse than reusing the old temperature; this experiment does **not** establish that the new temperature improves every calibration metric. Relative to uncalibrated FP16 probabilities (temperature 1), test log loss improved from 0.287197 to 0.273790. ECE is a descriptive, bin-dependent estimate.

The policy remains the one selected on development data. Test metrics were not used to choose another temperature or threshold.

## Development evidence and test isolation

The script reconstructs the original seed-42 grouped splits and verifies their hashes against the saved training report: 1,037 tuning, **466 calibration**, and **466 acceptance-selection** examples. Training, development and test sets are also checked for overlapping text and groups. The head is frozen; the tuning subset is not reused for this fit.

1. Encode the calibration and acceptance subsets on FP16 WebGPU, with masked mean pooling and normalization, using batches of 32 and the existing 256-wordpiece limit.
2. Fit a scalar temperature using the existing implementation. Calibration-subset log loss at temperature 1 was 0.299984, falling to 0.289552 at temperature 0.836752. This is fitting loss, not independent evidence of generalization.
3. Use the existing fixed 13-threshold grid, one representative per group, and one-sided exact binomial bounds with the existing Bonferroni correction. Under the FP16 temperature, threshold 0.85 accepted 376 examples with 367 correct; its corrected lower bound was **94.64%**, below the 95% target. Threshold 0.90 accepted 360 with 357 correct, giving a **96.89%** corrected lower bound. It was the highest-coverage qualifying candidate.
4. Write `policy.json`, freeze it, and load it back from disk before exposing test inputs to the browser. Refitting within that experiment is then blocked. The browser uses the saved policy for predictions; the server independently verifies every class, acceptance decision and maximum probability against the submitted logits and frozen policy.
5. Evaluate all test examples in batches of 32 and again individually. Neither result changes the saved policy.

The complete [recorded report](../benchmarks/browser-fp16-calibration.json) includes hashes, all threshold candidates, reliability bins and both evaluation modes. Statistical bounds assume representative, independent observations; they do not cover distribution shift.

Revalidated on 2026-09-21 after retraining the format-v3 bundle: the head, selected calibration, and reported test results were unchanged. The saved export now references the retrained bundle.

## Batch stability

The frozen policy accepted 2,394 inputs with 2,362 correct in batches, versus 2,396 with 2,363 correct individually. The per-mode one-sided 95% test lower bounds were **98.21%** and **98.16%**, respectively. These test bounds are descriptive and were not used for threshold selection.

Across all test examples, changing batch size changed **3 winning labels and 2 acceptance decisions**. Maximum confidence differed by at most 0.008672. Both modes met the requested accuracy target on this test set, but predictions are not bit-identical across batch shapes. The browser SDK should not promise exact batch/single equivalence for this FP16 path.

## Reproduce and use the artifact

Prepare the BANKING77 model and pinned FP16 assets using the [browser benchmark instructions](browser-benchmark.md), then run from the checkout:

```sh
npm run build
node --test scripts/browser-calibration/analysis.test.mjs
node scripts/calibrate-browser.mjs
```

Open the printed loopback URL and select **Calibrate and validate**. All encoding stays in the browser; the local Node process fits the scalar temperature and selects the threshold using the existing training logic. No teacher service is called, no input is uploaded, and no new model download occurs during this run.

Each run creates a new directory under `models/browser-fp16-calibration/` with `policy.json`, `report.json`, and development/test logits for auditing. An optional positional output directory is supported and must not already exist. Stop the local server with Ctrl-C after completion.

`policy.json` is an **experimental deployment-policy artifact**, tied to the head, encoder, tokenizer, data and runtime provenance. It is separate from the format-v3 model bundle and is loaded by the browser SDK, not the Node SDK. Generate it against the exact bundle you deploy; retraining an older model requires a new matching calibration export. Keep its FP16 policy with the FP16 export; the q8 fallback retains its own calibration. The original q8 model and report remain unchanged.

# Automatic training

Use the existing command and SDK:

```sh
node dist/cli.js train --task task.json --data examples.jsonl --out model-dir
```

Training recommends a probability cutoff targeting 95% accuracy against the supplied labels among predictions above it. `--target-accuracy 0.9` changes that target. The SDK returns answers and probabilities without accepting or rejecting them. Read `classifier.metadata.thresholdRecommendation` or use `jimothy inspect --model model-dir`; the CLI training output and both bundle JSON files expose the same recommendation.

## Data separation

Training first reserves approximately 20% as development data, unless `--validation` supplies it explicitly. The development pool is divided by grouped, stratified, deterministic splitting into approximately 50% model selection, 25% calibration, and 25% acceptance selection. A related group stays in exactly one subset. Classes with fewer than three independent development groups remain in model selection; their rows are never reused for calibration or acceptance selection.

The optional `--test` set is checked for input and group overlap with all training/development data, then evaluated only after the model, calibration and recommendation are frozen. No test results choose regularization, temperature, or threshold. Without a test set, summary metrics are marked as development results and are not an independent final evaluation.

## What happens internally

1. Compute frozen MiniLM embeddings once, or fit TF-IDF on training data only and encode each subset once.
2. Fit the head with L2 strengths `0.001, 0.0001, 0.00001, 0.000001, 0`. The default is at most 200 epochs per candidate, with early stopping after 30 epochs without validation-loss improvement. Choose the candidate and epoch with the lowest model-selection cross entropy against the supplied training distributions. `--l2` bypasses the search for controlled experiments.
3. Freeze the head and fit a temperature between 0.05 and 20 by minimizing log loss against winning reference labels on the calibration subset. At least 30 examples are required.
4. Evaluate a fixed threshold grid on the independent acceptance subset. For each threshold, count correct accepted predictions and compute a one-sided exact binomial lower bound, with a Bonferroni correction for the 13 thresholds searched. Choose the highest coverage with at least 30 accepted independent examples and a lower bound meeting the target. If none qualifies, store a null recommendation with status `insufficient_data` or `target_not_met`. Selection evaluates a cutoff on maximum class probability alone, with no hidden rejection rules.
5. Export the selected head, temperature, advisory threshold, provenance, and detailed report. Evaluate the optional test set with the same calibrated probabilities as the SDK, reporting hypothetical coverage and accuracy above the recommended cutoff. These metrics are null when no recommendation exists; the coverage curve is still available.

The threshold grid is `0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.925, 0.95, 0.975, 0.99, 0.995, 1`. The family-wise confidence level is 95%. This deliberately favors adequate evidence over optimistic coverage. For example, a handful of correct predictions is not enough to support a 95% accuracy target.

When group IDs are supplied, acceptance evidence counts only one deterministic representative per group. The binomial bounds assume independent, representative observations; group IDs do not establish independence or protect against distribution shift. Summary metrics still count all examples, so the report retains representative-level threshold evidence separately. Rare-class quality and out-of-scope detection need additional evaluation.

## Recommendations and application policy

`thresholdRecommendation` contains `threshold`, `status`, and `targetAccuracy`. A `ready` recommendation chooses the highest-coverage qualifying cutoff from the grid, breaking ties in favor of the lowest cutoff. For Score tasks, accuracy means the winning level matches the reference, not an error bound on the expected score.

`report.json` retains all candidates under `thresholdSelection`, including counts, coverage, observed accuracy and corrected lower bounds. Small datasets still produce usable prediction bundles: they return a null cutoff and explain why. No prediction is suppressed, even at a probability of 1 or with no known TF-IDF features.

Any acceptance policy belongs to the caller:

```ts
const prediction = await classifier.predict(input);
const { threshold } = classifier.metadata.thresholdRecommendation;
const accepted = threshold !== null && prediction.maxProbability >= threshold;
```

Detailed reports preserve human and teacher provenance. The default accuracy has one meaning: agreement with the winning supplied reference label. Soft distributions remain the head's training target, while probability calibration fits the winning reference label. Consequently, calibrated outputs may differ from the teacher's original probabilities; they estimate reference-label agreement rather than promise exact reproduction of the teacher's uncertainty.

## Sources

- [Guo et al., On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html): temperature scaling.
- [Scikit-learn calibration documentation](https://scikit-learn.org/stable/modules/calibration.html): disjoint calibration data and probability assessment.

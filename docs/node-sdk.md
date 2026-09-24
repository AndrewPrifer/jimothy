# Node.js SDK and model bundles

Install with `npm install jimothy`. Requires Node.js 22 or later. Load a bundle produced by the [training CLI](cli.md). For browser usage, see [the browser SDK](browser-sdk.md).

```ts
import { loadClassifier } from 'jimothy';

const classifier = await loadClassifier('./models/support');
try {
  const prediction = await classifier.predict('I was charged twice');

  console.log(prediction.answer, prediction.maxProbability);

  // Optional application policy. The SDK makes no acceptance decision.
  const { threshold } = classifier.metadata.thresholdRecommendation;
  const accepted = threshold !== null && prediction.maxProbability >= threshold;

  const batch = await classifier.predictBatch(['Where is my parcel?', 'The app crashes']);
  const result = await classifier.evaluate({ state: 'My invoice is wrong' });
  // result.answers.route: Jev-compatible typed answer fields
} finally {
  await classifier.dispose();
}
```

`loadClassifier` verifies the manifest, shapes, and all bundled asset checksums before inference. Tokenizers and ONNX files load from the bundle with `local_files_only: true`. No API keys, network services, runtime downloads, or hosted embeddings are required. Keep all files in the exported directory together. Checksums detect corruption; they are not signatures establishing who produced a model.

## API

| Call | Result |
| --- | --- |
| `await loadClassifier(directory)` | A classifier loaded from a complete local bundle. The Node loader takes no options. |
| `await classifier.predict(state)` | `{ answer, maxProbability }`. |
| `await classifier.predictBatch(states)` | Predictions in input order; an empty array returns an empty array. |
| `await classifier.evaluate({ state })` | `{ model, answers: { [questionId]: answer } }` for the bundle's fixed question. |
| `classifier.metadata` | A copy of `{ id, task, backend, thresholdRecommendation, createdAt }`. |
| `await classifier.dispose()` | Releases the encoder. Repeated disposal is safe; subsequent predictions reject. |

States are non-empty strings, JSON objects, or JSON arrays. Object keys are recursively sorted; array order is preserved. Use the same input shape and fields as training. All inputs have a 100,000-character limit, plus the backend token limit below. Overlength inputs reject rather than truncate.

| Answer type | Returned `answer` |
| --- | --- |
| `choice` | `{ type: 'choice', choice, probabilities }`, keyed by all named labels. |
| `boolean` | `{ type: 'boolean', probability }`, where `probability` is the probability of true. |
| `noul` | `{ type: 'noul', noul }`, where `noul` is the probability of true. |
| `score` | `{ type: 'score', score, probabilities }`, keyed by zero-based criterion indices; `score` is the distribution's expected index. |

`maxProbability` is the winning class probability. For binary answers it is `max(p, 1 - p)`; for Score it is the probability of the most likely level, not the expected score. The task definition is stored in the bundle and cannot be changed at prediction time.

## Threshold recommendations

Training recommends a cutoff on `maxProbability`, aiming for **95% accuracy against the supplied labels among predictions above that cutoff**. `train --target-accuracy 0.9` changes the target. The selected cutoff maximizes coverage among the fixed grid candidates supported by independent validation evidence. It is task- and runtime-specific, not a universal optimum or a guarantee on future inputs.

Read the same recommendation from `classifier.metadata.thresholdRecommendation`, `model.json`, `report.json`, or `jimothy inspect --model model-dir`:

```json
{ "threshold": 0.85, "status": "ready", "targetAccuracy": 0.95 }
```

When evidence is insufficient, `threshold` is `null` with status `insufficient_data` or `target_not_met`. Predictions are still returned, including for TF-IDF inputs without known features. Applications may use the recommendation, choose their own cutoff, or ignore it. The report's `thresholdSelection` includes candidate cutoffs, coverage, observed accuracy, and statistical lower bounds; it does not impose a runtime policy. The recommendation concerns the winning class, including for Score questions, rather than a bound on score error.

`predict` returns `{answer, maxProbability}`; `evaluate` returns `{model, answers}`. There is no `accepted`, `reason`, or `decisions` field, and no `--threshold` flag or threshold option. The Node loader takes only a directory; browser options control asset loading and execution. Jev's derived `confidence` field is deliberately not reproduced with different semantics.

Both SDKs require format version 3, including calibration and advisory threshold recommendations. Older bundles are unsupported; retrain them with the current CLI.

The Node SDK uses ONNX Runtime through Transformers.js for MiniLM; the linear head runs in JavaScript. TF-IDF inference does not import Transformers.js. An AI SDK provider adapter is not implemented yet.

## Architecture and input limits

MiniLM is the default; if `--encoder` is omitted, training downloads and bundles its public assets. The encoder is the q8 ONNX export of `Xenova/all-MiniLM-L6-v2`, with masked mean pooling and normalisation. Its immutable upstream revision and asset checksums are saved in the bundle. Inputs longer than **256 wordpieces, including special tokens, are rejected**, not truncated. It is intended for short English text.

`train --long-input chunk` opts into non-overlapping tokenizer windows for overlength MiniLM inputs. Each window is embedded with the usual mean pooling and normalization; the window vectors are averaged and normalized into one feature vector. The mode is recorded in `model.json` and used by both Node and browser inference. Inputs within the limit keep their existing embedding path. Each extra window adds encoder work; evaluate latency and model quality separately for this mode.

TF-IDF uses lowercase, Unicode-normalised word unigrams and bigrams, sublinear term frequencies, smoothed inverse document frequencies, and L2 normalisation. Its vocabulary is learned from the training split only. It supports up to 4,096 word tokens per input. MiniLM and TF-IDF both train a linear softmax head with full-batch Adam and soft-target cross entropy. Encoder fine-tuning is not implemented in v0.1.

## Evaluation and reproducibility

```sh
npx jimothy validate --task task.json --data examples.jsonl
npx jimothy evaluate --model models/support --data test.jsonl
```

Without `--validation`, training reserves approximately 20% of the input examples as development data. It divides development data into separate subsets for model selection, probability calibration, and threshold selection, keeping related groups together. Exact identical inputs are deduplicated; conflicting targets or groups for the same input are rejected. Explicit validation/test files are checked for input and group overlap. Near duplicates require user-supplied groups; the tool does not detect semantic duplicates automatically.

Features are computed once. The trainer searches five regularization strengths, selects the head and epoch by validation loss, fits a single temperature on calibration data, and recommends a threshold using the separate threshold-selection subset. TF-IDF fitting sees training inputs only. Supply `--test` for a separate test set evaluated after all these choices are frozen. Evaluation on a later file does not prove that file is independent of training.

The default CLI output includes accuracy against the supplied labels and the threshold recommendation. Coverage and accepted accuracy describe what would happen if an application applied that recommendation; both are `null` when no cutoff is recommended. The same workflow applies to human and teacher labels; provenance stays in the detailed report. See [automatic training](automatic-training.md) for the selection protocol and its statistical assumptions.

Each bundle contains:

```text
model.json           Task, features, head, calibration, threshold recommendation, hashes
report.json          Split fingerprints, training settings, metrics, and limitations
encoder/             MiniLM only: ONNX model, tokenizer, source metadata, licence
```

Training examples are not copied into the bundle. TF-IDF vocabulary can reveal words from training inputs; treat models and reports as potentially sensitive artifacts. Bundle size is distinct from installed runtime size and process memory usage.

Detailed reports retain label provenance, per-class results, confusion matrices, tuning trials, calibration loss, and threshold evidence. A supplied teacher label is a reference for agreement, not independent ground truth. Temperature calibration fits agreement with the winning supplied label; it can change the answer probabilities and a Score answer's expected value while preserving the winning class. The CLI's minimum of 10 unique examples is a usability check, not evidence of sufficient data for a threshold recommendation.

The vocabulary and training are deterministic for a fixed dataset/order, backend, runtime, and seed. The report stores dataset fingerprints and the actual teacher versions present. Cross-platform floating-point differences can still affect neural embeddings. Outputs are written through a staging directory and existing model directories are refused.

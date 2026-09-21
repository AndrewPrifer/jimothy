# Larger datasets for the local classifier

Start with BANKING77. Its closely related banking intents exercise semantic distinctions, and its human reference labels let us measure correctness separately from imitation of a teacher.

| Dataset | Size | What it tests |
| --- | --- | --- |
| [BANKING77](https://github.com/PolyAI-LDN/task-specific-datasets) | 13,083 English queries; 77 intents | Fine-grained classification and a first meaningful architecture benchmark. CC BY 4.0. |
| [CLINC150](https://github.com/clinc/oos-eval) | 23,700 English queries including out-of-scope examples; 150 supported intents | Broader intent routing and whether the model can reject unsupported requests. CC BY 3.0. |
| [MASSIVE 1.1](https://github.com/alexa/massive) | More than 1 million utterances across 52 languages; 60 intents | Multilingual evaluation and larger ingestion workloads. CC BY 4.0. |

MASSIVE is a parallel translation/localization dataset, not a million independent English examples. Keep translated versions of a source utterance together when splitting it. Our current MiniLM encoder is intended for English; multilingual use needs a suitable encoder and separate evaluation. CLINC150's out-of-scope examples should be evaluated explicitly: a threshold on the current softmax is not an established out-of-domain detector.

## Prepare BANKING77

From the repository root, with Python 3 available:

```sh
python3 scripts/prepare-banking77.py
```

This downloads the official CSVs, category list, licence, and attribution from a pinned upstream revision, verifies SHA-256 checksums, and creates `datasets/banking77/`. Add `--offline` to require already-cached source files. The source cache is `.cache/banking77-source/`. Existing output directories are refused; use `--out` for a separate conversion.

Prepared splits:

| Split | Examples |
| --- | ---: |
| Training | 8,023 |
| Validation | 1,969 |
| Test | 3,079 |

All 77 classes occur in each split. We remove four duplicate training records, one duplicate test record, and seven training inputs that also occur in the official test set. Duplicate comparison uses Unicode NFKC, case folding, and collapsed whitespace; actual model inputs retain the original text. Validation is a deterministic, stratified 20% of the remaining official training set. This is a cleaned local benchmark, not the unmodified published BANKING77 evaluation protocol.

The output includes:

- `task.json`: one Jev-compatible Choice question with all 77 intent labels.
- `train.jsonl`, `validation.jsonl`, `test.jsonl`: inputs plus original human `label` fields, ready for supervised training and evaluation.
- `*.inputs.jsonl`: inputs with stable IDs and duplicate-group IDs, without labels.
- `*.outputs.jsonl`: matching IDs and the original labels represented as hard Choice answers.
- `manifest.json`, `excluded.json`, `LICENSE.txt`, `README.md`: provenance, checksums, exclusions, licence, and attribution.

The converted outputs are human annotations in a compatible shape. They are not Jev responses and contain no invented confidence scores. The task's criteria descriptions simply expand category names; review and disambiguate those descriptions before asking a teacher to label the inputs.

## Run the benchmark

```sh
npm ci
npm run build

# Run once if the encoder is not already cached:
node dist/cli.js prepare-encoder --out .cache/minilm

node scripts/benchmark-banking77.mjs
```

The runner verifies that all 8,023 separate input/output pairs import to the same inputs and training targets as the human-labelled rows. It trains both TF-IDF and frozen MiniLM with the human-labelled files and exports loadable SDK bundles to `models/banking77-auto-v2/{tfidf,minilm}`. To repeat without overwriting results, pass a new output directory as the first argument.

Both runs use automatic regularization tuning, temperature calibration, and acceptance selection with a 95% accuracy target. The 1,969 development examples become 1,037 tuning, 466 calibration, and 466 acceptance-selection examples. Each head candidate has at most 100 epochs, learning rate 0.05, and seed 42. TF-IDF has at most 4,096 features; MiniLM has 384. See the [automatic training protocol](automatic-training.md). The test set is evaluated only after these choices are frozen; it is the same previously inspected benchmark test set, not a fresh blind evaluation.

`models/banking77-auto-v2/results.json` records accuracy, macro F1, threshold coverage, bundle size, training time, load time, hardware, and warm single-input latency. Latency uses 20 warm-up calls and then 200 sequential SDK predictions on inputs spaced across the test set. It includes tokenization, feature extraction, and the head, but excludes model loading. Bundle size includes reports and assets; it excludes installed runtime dependencies and process memory.

## Automatic training results

The [automatic run snapshot](../benchmarks/banking77-auto-v2.json) records the complete settings and selection evidence. Run on the same Apple M3 Max, macOS arm64, Node 25.6.0, on 2026-09-20:

| Backend | Overall accuracy | Accepted fraction | Accuracy on accepted inputs | Warm SDK p95 | Bundle |
| --- | ---: | ---: | ---: | ---: | ---: |
| TF-IDF + linear head | 82.20% | 22.80% | 99.57% | 0.044 ms | 6.64 MB |
| Frozen MiniLM + linear head | 92.37% | 82.23% | 97.95% | 3.81 ms | 24.58 MB |

Both runs selected L2 = 0 at 100 epochs. TF-IDF selected temperature 0.7753 and threshold 0.995; MiniLM selected temperature 0.8231 and threshold 0.85. Total training/calibration/evaluation time was 37.6 seconds and 446.0 seconds, respectively. Searching several heads increases one-time training cost; inference still uses one head and one encoder. These capped runs do not establish that 100 epochs or zero regularization is optimal for other datasets.

On this benchmark, MiniLM's overall accuracy increased from 80.51% to 92.37%. About 82% of inputs clear its recommended cutoff, compared with zero under the original threshold. These are observed results on the previously inspected BANKING77 test set, not a guarantee for new inputs. Regularization and training selection account for changes in the winning labels; temperature calibration itself preserves them. The updated runner also verifies that reloaded SDK predictions reproduce the reported accuracy and counts above the recommended cutoff on the complete test set, with the cutoff applied in benchmark code.

## Initial baseline before automatic training

Run on 2026-09-20, Apple M3 Max, macOS arm64, Node 25.6.0, using the earlier fixed-L2 trainer. The portable [historical results snapshot](../benchmarks/banking77-v1.json) includes exact settings, source and split provenance, and coverage curves. Its bundles remain in `models/banking77-v1/`. The current runner uses the new automatic workflow and does not reproduce these historical settings.

| Backend | Test top-choice accuracy | Macro F1 | Warm SDK p95 | Bundle size | Training/evaluation time |
| --- | ---: | ---: | ---: | ---: | ---: |
| TF-IDF + linear head | 69.47% | 66.77% | 0.044 ms | 7.24 MB | 5.5 s |
| Frozen MiniLM + linear head | 80.51% | 78.63% | 3.30 ms | 24.51 MB | 94.6 s |

Accuracy above measures the top choice on every test input, before abstention. Both models accepted zero test examples at the old default 0.8 probability threshold. At the predeclared 0.5 reporting threshold, MiniLM accepted 157/3,079 examples (5.10% coverage), of which 156 were correct. That small accepted subset did not establish a general 99% accuracy guarantee, and 0.5 was not selected as a deployment threshold.

These results exposed a problem with carrying the prototype's fixed regularization and acceptance settings into a 77-class problem. No tuning or probability calibration was performed for this historical run. These are initial local baselines, not a claim of best achievable BANKING77 performance or a measured speed comparison against Jev.

The supervised benchmark validates the larger-data import, training, packaging, and local inference path. It does not yet test learning from a teacher's soft distributions.

## Teacher-labelled experiment

Use the same splits for a controlled comparison:

1. Keep the current human-label run as the supervised baseline.
2. Give a permitted teacher `task.json` and the `state` values from `train.inputs.jsonl`. Save its full answer distributions under the same IDs. Record model version, exact task, and generation settings.
3. Train from those input-only rows and the teacher response file. Keep `validation.jsonl` and `test.jsonl` as human-labelled evaluation files.
4. Compare human test accuracy, macro F1, calibration, and latency against the baseline. On a separately labelled held-out set, also measure agreement with the teacher.

For example, once `teacher.outputs.jsonl` exists:

```sh
node dist/cli.js train \
  --task datasets/banking77/task.json \
  --inputs datasets/banking77/train.inputs.jsonl \
  --outputs teacher.outputs.jsonl \
  --validation datasets/banking77/validation.jsonl \
  --test datasets/banking77/test.jsonl \
  --encoder .cache/minilm \
  --out models/banking77-teacher
```

Do not put the reference `label` into the teacher-training inputs: the importer deliberately lets human labels override teacher targets. Keep human gold data separately to avoid accidentally training the supervised baseline again. A hard-label benchmark tests architecture and format compatibility; it does not establish how well the model reproduces a teacher's probability distributions.

No teacher calls are made by these scripts. Jev-derived training data still requires terms permitting that use; see the [project README](../README.md) for the existing TypeSafe restriction.

Converter checks run without network access:

```sh
python3 -m unittest discover -s scripts/tests
```

New benchmark runs export format-v3 bundles with advisory thresholds. The recorded v2 results above are unchanged; acceptance is now an explicit benchmark policy outside the inference SDK.

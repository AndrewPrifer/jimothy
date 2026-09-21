# Jimothy

Train small, local classifiers from Jev-compatible examples or a teacher model. One npm package includes the training CLI, Node.js SDK, and browser SDK.

<p align="center">
  <img height="280" alt="addtext_com_MjMxNDI0Ond3dzEwOjQ0MDk5" src="https://github.com/user-attachments/assets/b0781bef-61ab-47e5-96c6-2c034ad5cae3" />
</p>

## Quick start

Requires Node.js 22 or later. Install and train a support-ticket classifier using the included examples:

```sh
npm install jimothy

npx jimothy train \
  --task node_modules/jimothy/examples/task.json \
  --data node_modules/jimothy/examples/training.jsonl \
  --out models/support

npx jimothy predict --model models/support --text "I was charged twice"
```

This uses the default MiniLM backend, which downloads and bundles its pretrained encoder during training. Keep the entire output directory together when moving or deploying a model.

## CLI

For your own task, provide a [task definition and dataset](docs/data-format.md). Datasets accept JSONL or JSON arrays. Choose one training form:

```sh
# Saved inputs and answers in one file
npx jimothy train --task task.json --data examples.jsonl --out models/custom

# Separate inputs and answers, joined by matching unique string IDs
npx jimothy train --task task.json --inputs inputs.jsonl --outputs outputs.jsonl --out models/custom

# Input-only data: generate answers with a teacher, then train
npx jimothy train --task task.json --inputs inputs.jsonl --teacher typesafe-ai/jev --out models/custom
```

The teacher form requires `AI_GATEWAY_API_KEY` in your environment and input rows such as `{"id":"1","state":"I was charged twice"}`. It sends inputs to the teacher and caches responses in `<out>.teacher/`; rerun the same command to resume. The other forms train locally from saved answers.

Training tunes regularization, calibrates probabilities, and recommends a cutoff. Add `--validation dev.jsonl` and `--test test.jsonl` for separate labeled evaluation sets; otherwise, training reserves a development split. Use `--backend tfidf` for a lightweight baseline or `--encoder path/to/encoder` to reuse MiniLM assets offline. Output directories must not already exist.

```sh
npx jimothy validate --task task.json --data examples.jsonl
npx jimothy predict --model models/custom --text "A new message"
npx jimothy predict --model models/custom --state '{"subject":"Receipt","body":"Paid"}'
npx jimothy predict --model models/custom --data inputs.jsonl
npx jimothy evaluate --model models/custom --data test.jsonl
npx jimothy inspect --model models/custom
npx jimothy --help
```

Prediction accepts exactly one of `--text`, `--state` (JSON), or `--data` (rows containing `state`). Results go to stdout as JSON; progress goes to stderr. File prediction emits one result per row. See the [CLI reference](docs/cli.md) for all options, encoder preparation, and custom teachers.

## Node.js SDK

```ts
import { loadClassifier } from 'jimothy';

const model = await loadClassifier('./models/support');
try {
  const prediction = await model.predict('I was charged twice');
  // { answer, maxProbability }
  const batch = await model.predictBatch(['Where is my parcel?', 'The app crashes']);
  const response = await model.evaluate({ state: 'My invoice is wrong' });
  // { model, answers: { [questionId]: answer } }
  console.log(model.metadata.thresholdRecommendation);
} finally {
  await model.dispose();
}
```

States may be strings, JSON objects, or arrays; use the same shape as training. Answers support `choice`, `boolean`, `noul`, and `score`. Node inference needs no network access. Reuse a loaded model across predictions, then dispose it when finished.

`metadata.thresholdRecommendation` contains `{ threshold, status, targetAccuracy }`. A `null` threshold means no recommendation is available, as with the small quick-start dataset. The SDK returns every prediction; applying a cutoff is your application's choice. See [SDK behavior and model bundles](docs/node-sdk.md) and [automatic training](docs/automatic-training.md).

## Browser SDK

Copy runtime assets and the complete trained model into your web app's public directory:

```sh
npx jimothy prepare-browser --out public/jimothy
mkdir -p public/models
cp -R models/support public/models/support
```

```ts
import { loadClassifier } from 'jimothy/browser';

const model = await loadClassifier('/models/support/');
try {
  const prediction = await model.predict('I was charged twice');
  console.log(prediction.answer, prediction.maxProbability);
} finally {
  await model.dispose();
}
```

Serve over HTTPS or localhost; `public/` maps to `/` in these examples. The browser uses the same prediction methods and metadata as Node, with inference in a worker. TF-IDF runs in JavaScript; MiniLM defaults to q8 WASM. FP16 WebGPU requires a matching calibrated export. MiniLM probabilities can differ across runtimes, so validate cutoffs in your deployment environment.

The [browser guide](docs/browser-sdk.md) covers custom asset URLs, WebGPU, loading progress, cancellation, and plain HTML usage.

## Speed

Recorded on an Apple M3 Max with Node.js 25.6.0. Inference is warm, single-input median latency, excluding model loading:

| Workload | Local training, tuning and evaluation | Inference |
| --- | ---: | ---: |
| BANKING77, TF-IDF | 37.6 s | 0.023 ms |
| BANKING77, MiniLM | 7 min 26 s | 1.72 ms |
| Email classifier (300 samples), MiniLM | 3.80 s | 6.80 ms |

**Input length strongly affects inference speed.** Longer emails require MiniLM to process more tokens than short BANKING77 queries, helping explain the higher email latency.

BANKING77 used 8,023 training inputs, 77 classes, and a 100-epoch cap per candidate; the CLI defaults to 200. Training times exclude downloads and teacher calls. In Chromium on the same machine, MiniLM took about **7.5 ms with WASM** and **5.6–8.7 ms with FP16 WebGPU** per warm input across the recorded runs. Results depend on the task, hardware, runtime, and caches. See [full measurements and methodology](docs/performance.md).

On BANKING77, **TF-IDF trained about 12× faster and predicted about 75× faster**, while **MiniLM achieved higher accuracy: 92.37% versus 82.20%**. In the browser, FP16 WebGPU's largest benefit was batch throughput—about 14–36× q8 WASM in these runs—with larger weights and higher initial startup cost.

## Development

From a source checkout:

```sh
npm ci
npm run build
npm run typecheck
npm test
```

Use `node dist/cli.js` in place of `npx jimothy`; fixture paths are under `examples/`. Core tests need no network or model downloads. To test real MiniLM inference:

```sh
node dist/cli.js prepare-encoder --out .cache/minilm
JEV_DISTILL_ENCODER=.cache/minilm npm run test:semantic
```

Encoder preparation downloads public weights; skip it if `.cache/minilm` already exists. Browser examples: [banking playground](examples/browser/README.md) (`npm run demo:browser`) and [email comparison with Jev](examples/email-browser/README.md) (`npm run demo:email`). Each guide explains the required model setup. More examples: [BANKING77](docs/datasets.md) and [email classifier (300 samples)](docs/email-classifier.md).

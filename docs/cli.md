# CLI reference

Install with `npm install jimothy` and run `npx jimothy`. Requires Node.js 22 or later. From a built source checkout, `node dist/cli.js` runs the same CLI. Use `npx jimothy --help` or `npx jimothy --version` for help and version information.

Commands below use your own files in the current directory. The [data format guide](data-format.md) defines task JSON, combined examples, and separate input/output rows. Installed sample files are under `node_modules/jimothy/examples/`; source-checkout samples are under `examples/`.

## Train

Choose exactly one source of answers:

```sh
npx jimothy train --task task.json --data examples.jsonl --out models/custom
npx jimothy train --task task.json --inputs inputs.jsonl --outputs outputs.jsonl --out models/custom
npx jimothy train --task task.json --inputs inputs.jsonl --teacher typesafe-ai/jev --out models/custom
```

These are alternatives, not sequential commands. Training refuses an existing output directory. Combined and separate files accept JSONL or JSON arrays; separate files join by unique string IDs. `--teacher` takes unlabeled inputs and cannot be combined with saved outputs or labels. All examples must describe the same selected question.

| Option | Meaning / default |
| --- | --- |
| `--task FILE` | Jev-compatible question definition. Optional if the first input/request embeds questions. |
| `--question ID` | Select one question when the task has several. |
| `--data FILE` | Combined inputs and answers. Mutually exclusive with separate `--inputs`/`--outputs`. |
| `--inputs FILE` | Input rows; use with `--outputs` or `--teacher`. |
| `--outputs FILE` | Saved answers matched to `--inputs` by ID. |
| `--out DIR` | Required new model output directory. |
| `--backend minilm\|tfidf` | `minilm` by default; TF-IDF needs no encoder assets. |
| `--encoder DIR` | Copy a previously prepared MiniLM encoder. Otherwise download pretrained assets. |
| `--validation FILE` | Labeled development data; otherwise reserve approximately 20% of the supplied data. |
| `--test FILE` | Separate labeled data, evaluated after model selection and calibration. |
| `--target-accuracy N` | Accuracy target for the advisory probability cutoff; default `0.95`. |
| `--epochs N` | Maximum epochs per candidate; default `200`. Early stopping can shorten training. |
| `--learning-rate N` | Adam learning rate; default `0.05`. |
| `--l2 N` | Fix regularization instead of searching five strengths. Calibration still runs. |
| `--max-features N` | TF-IDF vocabulary limit; default `4096`. |
| `--seed N` | Grouped split seed; default `42`. |
| `--teacher MODEL` | Model used to generate answers before training. |
| `--teacher-url URL` | Complete TypeSafe-compatible endpoint; default Vercel AI Gateway. |
| `--teacher-key-env NAME` | Environment variable holding the API key; default `AI_GATEWAY_API_KEY`. |
| `--teacher-cache DIR` | Resumable label cache; default `<out>.teacher/`. |
| `--teacher-rpm N` | Optional cap on request starts per minute; no default cap. |

The output contains the model directory and ID, backend, report path, summary metrics, threshold recommendation, and warnings. The [automatic training guide](automatic-training.md) explains splitting, regularization search, probability calibration, and recommendation selection. [Model bundles](node-sdk.md) covers architecture, input limits, reports, and the bundle format.

## Validate, predict, evaluate and inspect

```sh
npx jimothy validate --task task.json --data examples.jsonl
npx jimothy validate --task task.json --inputs inputs.jsonl --outputs outputs.jsonl

npx jimothy predict --model models/custom --text "I was charged twice"
npx jimothy predict --model models/custom --state '{"subject":"Receipt","body":"Paid"}'
npx jimothy predict --model models/custom --data inputs.jsonl

npx jimothy evaluate --model models/custom --data held-out.jsonl
npx jimothy inspect --model models/custom
```

`validate` imports and checks the dataset without training or calling a teacher. It also accepts `--question` and reports the selected task, unique example count, duplicates removed, and label provenance counts.

`predict` requires exactly one of `--text`, `--state`, or `--data`. `--state` is JSON, not a path. Prediction files contain rows with `state` and an optional `id`. Use the same state shape as training. Each result has `{model, answers}`; file predictions also preserve `id` when supplied. The selected task is fixed in the model, so prediction takes no task or question argument.

`evaluate` requires a combined labeled file in `--data`. It reports accuracy against the supplied labels, the advisory cutoff, hypothetical coverage above that cutoff, and batch inference time. `inspect` returns the stored task, backend, recommendation, and full report without predicting. Neither command calls a provider or downloads encoder assets.

Results are JSON on stdout; progress and errors go to stderr. File prediction emits one JSON result per input row. Errors exit nonzero. Options are specific to each command: for example, `--teacher` is only valid for `train`, and there is no `--threshold` option.

## Prepare assets

To download a reusable encoder once and train offline from saved examples:

```sh
npx jimothy prepare-encoder --out .cache/minilm
npx jimothy train --task task.json --data examples.jsonl --encoder .cache/minilm --out models/custom
```

`prepare-encoder` downloads only public pretrained assets. Pass its directory to `--encoder`; a model's existing `encoder/` directory also works. Training copies those files into the new bundle, which can then run independently of the source directory.

For a browser app:

```sh
npx jimothy prepare-browser --out public/jimothy
```

This copies the installed browser worker and runtime assets without network access. Install optional dependencies to use this command. Both preparation commands require a new output directory and accept only `--out`. See the [browser SDK guide](browser-sdk.md) for model hosting and loading.

## Train with a teacher

Set `AI_GATEWAY_API_KEY` in your environment, then provide a task and unlabeled inputs. Each input needs a unique string `id` and a `state`:

```json
{"id":"email-1","state":{"from":"shop@example.com","subject":"Your order shipped","body":"Your order is on its way."}}
```

```sh
npx jimothy train \
  --task task.json \
  --inputs emails.jsonl \
  --teacher typesafe-ai/jev \
  --out models/email
```

This sends the selected question and each unique input state to the teacher, saves its Jev-compatible responses, and runs the normal training pipeline. Question definitions embedded in the first input/request also work. Existing `--data` and `--inputs`/`--outputs` workflows are unchanged; `--teacher` cannot be combined with existing outputs or labels. Explicit `--validation` and `--test` files must already contain labels and remain separate.

The default endpoint is Vercel's [TypeSafe-compatible API](https://vercel.com/docs/ai-gateway/sdks-and-apis/typesafe). For another compatible model or provider:

```sh
npx jimothy train --task task.json --inputs inputs.jsonl \
  --teacher your-model \
  --teacher-url https://your-provider.example/v1/systemone \
  --teacher-key-env YOUR_PROVIDER_API_KEY \
  --out models/your-task
```

`--teacher-url` is the complete **TypeSafe-compatible** POST endpoint accepting `{model,state,questions}` and returning `{model,answers}`. It is not an OpenAI chat-completions endpoint. Boolean questions are sent as native `noul`; Choice and Score probability distributions are retained. The key is read from the named environment variable, never a CLI argument or saved credential.

Completed responses are saved incrementally in **`<out>.teacher/`**, outside the inference bundle. Rerun the same command after an interruption. Four requests run concurrently; network failures, HTTP 429 and server errors receive up to two retries. A request that reaches the provider but loses its response may still be billed again on retry.

- `run.json` identifies the question, requested model, and endpoint. Changes require a new cache directory.
- `responses.jsonl` stores validated responses by input content, including resolved model versions and usage metadata.
- `outputs.jsonl` contains ID-matched outputs for the current input file after labeling completes; it can be passed to ordinary `--outputs` training.

For reproducibility, prefer a versioned teacher model when the provider supports one. An alias such as `typesafe-ai/jev` can change upstream even though the cache identity remains the same; response identifiers are recorded but may themselves be aliases.

Use `--teacher-cache DIR` to reuse responses across runs, input subsets, or output model directories. Cache-only runs need no API key. If your provider has a low rate limit, add `--teacher-rpm 20` to pace request starts; retries count toward this limit and HTTP 429 pauses new work too. Keep caches private: they contain teacher responses. Adding `--teacher` is the only training option that uploads your inputs; the SDK remains fully local.

The [300-email example](email-classifier.md) includes reproducible generation, teacher labeling, local training, and SDK measurements for `primary`, `promotion`, `update`, `social`, `forum`, and `purchase`.

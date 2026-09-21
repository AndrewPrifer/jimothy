# Task and dataset format

## Task definition

For example, save this as `task.json`:

```json
{
  "questions": {
    "route": {
      "type": "choice",
      "instructions": "Which team should handle this support request? Choose the primary issue.",
      "criteria": {
        "billing": "Payments, charges, invoices, subscriptions, and refunds",
        "shipping": "Delivery, parcels, tracking, and shipping addresses",
        "technical": "Application errors, login failures, crashes, and broken features"
      }
    }
  }
}
```

Provide a task JSON file containing `questions` (see [the included task](../examples/task.json)). One question is trained per bundle. For requests with several questions, select one with `--question route`. All examples must use the same selected instructions, criteria, and question type. Changing a task requires retraining.

## Combined records

Combined datasets accept **JSONL** or a **JSON array**. Supported records:

```json
{
  "id": "ticket-001",
  "group": "customer-42",
  "input": { "state": "My card was charged twice." },
  "output": {
    "model": "your-permitted-teacher-version",
    "answers": {
      "route": {
        "type": "choice",
        "choice": "billing",
        "probabilities": { "billing": 0.95, "shipping": 0.02, "technical": 0.03 }
      }
    }
  }
}
```

`request`/`response` are aliases for `input`/`output`. Flat `{state, answers}` rows also work. Questions can be present in each input/request; `--task` is optional if the first request includes them. Native TypeSafe and Vercel response metadata is accepted, and the model identifier is recorded when available. Teacher confidence is not used as a training weight.

## Separate files

Separate sets must each contain matching, unique **string `id` fields**:

```text
inputs.jsonl:  {"id":"1","state":"My card was charged twice."}
outputs.jsonl: {"id":"1","answers":{"route":{"type":"choice","choice":"billing"}}}
```

```sh
npx jimothy train --task task.json \
  --inputs inputs.jsonl --outputs outputs.jsonl --out models/my-task
```

The join is by ID, never row position. Missing, duplicate, and extra IDs are errors. Full probability distributions are preferred. Hard Choice answers are supported. Distribution keys must match the criteria exactly; sums within 0.01 of 1 are renormalised to account for rounding, and other invalid distributions are rejected.

## Answer types

Supported question types:

| Type | Criteria | Training answer |
| --- | --- | --- |
| `choice` | Object of 2–255 named labels and descriptions | `probabilities`, or a hard `choice` |
| `boolean` | Optional `true`/`false` descriptions | `probability` between 0 and 1 |
| `noul` | Optional `true`/`false` descriptions | `noul` between 0 and 1 |
| `score` | Ordered array of 2–10 level descriptions | Full level `probabilities`, keyed `"0"`, `"1"`, etc. |

Boolean and Noul answer spellings are accepted interchangeably on import. Score learns a distribution over levels; its returned score is their probability-weighted mean. A score mean by itself is insufficient for training this model.

## States and human labels

An optional row-level `label` is a **human-supplied label**. It overrides the teacher target for training and is tracked separately in metrics. Use a string for Choice, a boolean for Boolean/Noul, or an integer level for Score. Human-only rows such as `{"state":"Refund my payment", "label":"billing"}` are also supported. Do not mark machine-generated labels as human labels if you want meaningful human-accuracy reporting.

States may be strings, JSON objects, or JSON arrays. Object keys are recursively sorted into canonical JSON; arrays keep their order. The same preprocessing runs in training and inference. Select relevant fields before collecting examples. There is no automatic summarisation or field selection. All variable information that matters to the task must be available to the local model at prediction time.

## Dataset validation

Training requires at least 10 unique examples and at least two winning labels in the training split. These minimums do not establish model quality or sufficient calibration evidence. Validate imports before training:

```sh
npx jimothy validate --task task.json --data examples.jsonl
```

Exact duplicate inputs are deduplicated; conflicting targets are errors. Use a shared `group` value for related examples that must stay in the same split. Explicit validation and test sets are checked for overlapping inputs or groups. See [automatic training](automatic-training.md) for splitting and calibration, and [larger datasets](datasets.md) for benchmark data.

# Step-608 validation scoring fix

The generation file `outputs/validation-best-step608.jsonl` has 300 predictions
for 100 validation records. Its original content and provenance are unchanged.

## Causes and fixes

1. `enrich_rows` incorrectly required globally unique question IDs. Generated
   probe IDs such as `generated_irrelevant_0` are local to a record. The evaluator
   now joins and checks completeness using `(split, record_id, question_id)`.
   Duplicates within that identity, missing predictions and altered source fields
   are still rejected.
2. One probe in `squad_validation_31` was rescued as answerable, but has no
   authoritative answer reference. The evaluator verifies its generated provenance
   against the annotation data, retains the original labels/gold/prediction, and
   reports it explicitly as excluded. It does not invent gold or call it irrelevant.
   All 100 source answerable questions and 199 irrelevant questions are scored;
   one rescued probe is excluded from these metrics. Missing gold for an ordinary
   source question still raises an error.
3. Molab previously captured scoring stderr without displaying it. Captured
   subprocess failures now print the original error before re-raising.

## Local scoring artifacts

`outputs/evaluation-best-step608-local/` contains `metrics.json`,
`scored_predictions.jsonl`, `report.json` and a byte-normalized LF scoring copy
of the annotation data. The LF copy matches the dataset SHA-256 recorded in all
300 predictions exactly. The local original uses different line endings; neither
the original dataset nor recorded prediction fingerprints were changed.

| Metric | Result |
|---|---:|
| Source answerable F1 | 6.97% |
| Exact match | 3.00% |
| ROUGE-L | 7.30% |
| Refusal F1 | 80.38% |
| False refusal on source answerable questions | 51.00% |
| Gold recall@8 | 95.50% |

Scoring now completes, but answer generation quality is still low. High routing
recall@8 is not answer accuracy; many records have few atoms. No retraining or
new generation was performed, and no checkpoint metadata was modified.

The large training-report `examples[].prediction` strings are decoded tokenwise
argmax under teacher forcing (`training/loop.py`), not autoregressive generation.
Use the saved generation JSONL and its evaluation metrics for answer quality.

After deploying the updated code to Molab, click evaluation again. The notebook
will rescore its existing prediction file rather than regenerating it, provided
its checkpoint/dataset fingerprints still match. Do not edit those fingerprints
or rerun training merely to resolve this scorer error.

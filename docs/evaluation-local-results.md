# Local evaluation protocol and results (2026-10-25)

## What was verified

- `uv run --extra test pytest -q` covers SQuAD-style multi-reference F1/EM, separate ROUGE-L, refusal confusion counts, routing at K=1/4/8, one-line stopping, strict comparison, checkpoint artifacts, and a CPU toy `train → checkpoint → evaluate` smoke test.
- The smoke uses the real training checkpoint writer/loader and evaluation CLI, with a deterministic toy answer engine in place of Gemma generation. It confirms a valid config fingerprint loads and a changed fingerprint is rejected. It does **not** measure real model accuracy.
- The local best and last pointers refer to steps 31 and 36. These are from a different run than the supplied step-28 predictions, so they cannot form a controlled before/after inference comparison. The base model is not cached locally and there is no local GPU.

## Historical prediction re-score

Input: `D:\Downloads\predictions.jsonl` (10 validation predictions; 8 answerable, 2 generated irrelevant; checkpoint step 28). The existing file was read without regeneration or overwrite.

| Metric | Historical report | Current scorer on unchanged predictions |
| --- | ---: | ---: |
| Answerable F1 | 20.70% | 21.91% raw SQuAD-style F1 |
| ROUGE-L | 20.70% | 20.70% |
| First-line F1 | Not reported | 58.82% diagnostic only |
| Refusal F1 | 0% | 0% (TP=0, FP=1, FN=2, TN=7) |
| False refusal rate | Not reported | 12.5% of answerable questions |
| False answer rate | Not reported | 100% of irrelevant questions |
| Recall@8 | 100% reported | Not recomputed by score-only without validated atom join |

The F1 change is **scoring-only**: English SQuAD normalization removes punctuation and articles and takes the maximum over source answers. The first-line number shows the historical long-answer contamination, but is not substituted for the primary raw F1. There is no measured after-generation benchmark yet.

The local atomized JSONL has two invalid source spans. `doc2atom-evaluate --predictions ... --data ... --source ...` correctly refuses to join it, so this report does not claim validated Recall@1/4/8 or MRR for the historical file. Its reported Recall@8 also has weak discriminative value: that validation record has nine atoms and K=8 covers 88.9% of them.

## Re-run protocol on Molab after transferring this code

Use the exact training snapshot and its matching checkpoint and data; do not substitute the local step-31/36 checkpoint for the historical step-28 predictions. First repair/regenerate any atomized records with invalid source spans and run source validation. Keep `--wandb-mode disabled` for a local/offline check, or explicitly select `offline`/`online` when artifact logging is intended.

```bash
uv run doc2atom-validate --input data/squad_molab_atoms.jsonl --source data/squad_molab_raw.jsonl --reject-span-mismatch
uv run doc2atom-evaluate --config outputs/molab-single-gpu/config/resolved.yaml --data data/squad_molab_atoms.jsonl --source data/squad_molab_raw.jsonl --checkpoint outputs/molab-single-gpu/checkpoints/best.json --output outputs/molab-single-gpu/predictions/validation.jsonl --split validation --wandb-mode online
uv run doc2atom-compare --input doc2atom=outputs/molab-single-gpu/predictions/validation.jsonl --input baseline=outputs/baseline/predictions/validation.jsonl --output outputs/comparison/report.json
```

The compare command checks question IDs, source fields, gold answers, source/atomized fingerprints and generation/scoring settings. Legacy rows lacking provenance are labeled noncomparable. A baseline file must be produced under the same source dataset and protocol; merely changing the model name does not make an old file comparable.

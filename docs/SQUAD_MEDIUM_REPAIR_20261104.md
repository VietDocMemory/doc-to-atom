# Medium SQuAD annotation repair (2026-11-04)

## Result

`outputs/squad-medium-repaired-20261104/` passes source-aware validation:

| Metric | Before | After |
|---|---:|---:|
| Records | 600 | 600 |
| Source questions/answers preserved | 593/600 | 600/600 |
| Source-aware errors | 31 | 0 |
| Invalid source spans | 2 | 0 |
| Grounding failures | 10 | 0 |
| Parse failures | 14 | 0 |

The repair contains 5825 atoms and 1800 questions (600 source questions and
generated probes). Historical warning/transformation messages are preserved;
they do not represent unresolved validation failures.

574 valid records remain byte-identical. Original input, source and response
files remain unchanged. The separate bundle contains `atoms.jsonl`,
`raw_source.jsonl`, `raw_responses.jsonl`, retry-only responses, source-reviewed
repair plan, old/new records, exact API requests, before/after audits,
fingerprints, `verification.json` and `release.json`.

## Why blind retries did not help

The model repeatedly corrected source spelling/grammar in copied quotations:
`his` instead of source `her`, `procuring` instead of source `procurring`, and
`warszawicy` instead of source `warszawiacy`. Source spans must be copied
verbatim, including the source's own errors. The repair only accepts the reviewed
substitutions when the resulting complete span exists in the original context.

Other failures included literal unescaped ampersands and extra closing XML tags
in probe generation, missing opening delimiters in question annotations,
omitted date/attribution facts, and case/punctuation variants absent from the
chosen gold atom evidence. All source questions, answer variants and contexts
are preserved. Grounding policy and validation gates were not relaxed.

The explicit plan is `configs/annotation_repairs/squad_medium_20261104.json`.
It includes reviewed semantic corrections, not just XML syntax recovery; these
are identified as such in each record's transformations. Original provider
responses are never rewritten to appear corrected.

Only question annotation was retried for the 14 parse-failed records, using
MiniMax-M3.1-Flash, reasoning `low`, a 524288-token output cap, and no automatic
retry. Valid atom banks and saved probes were reused. Incremental usage was
14 requests / 30025 total tokens. Actual billed USD was not available.

## Using the repaired bundle in Molab

Upload the complete repair folder to:

```text
/marimo/doc-to-atom/outputs/squad-medium-repaired-20261104/
```

Use the updated `notebook_molab_merged.py`, select
**Repaired medium SQuAD (20261104)**, and click **2b. Validate existing raw + atoms**.
This selects the paired repaired files directly; the original `data/` files are
not overwritten. Upload both source and annotations, not atoms alone. The older
Molab files and the local repaired files may have different atom IDs after retries.

The updated prepare button stops if an existing cache is invalid. It never
silently regenerates paid annotations merely because validation failed.

Independent read-only validation:

```bash
uv run --env-file /marimo/.env python scripts/validate_data.py \
  --input outputs/squad-medium-repaired-20261104/atoms.jsonl \
  --source outputs/squad-medium-repaired-20261104/raw_source.jsonl \
  --reject-span-mismatch
```

## Reproducing the repair on a different cached copy

This command incurs paid question-stage requests. Do not rerun it on the already
validated bundle. It refuses existing output directories and unreviewed spans.

```bash
uv run --env-file /marimo/.env python scripts/repair_annotations.py \
  --source data/squad_molab_raw.jsonl \
  --input data/squad_molab_atoms.jsonl \
  --raw-responses data/squad_molab_atoms.jsonl.raw-responses.jsonl \
  --output-dir outputs/squad-medium-repaired-20261104 \
  --allow-api
```

No training, checkpoint metadata changes, GitHub push or Molab execution was
performed as part of this repair.

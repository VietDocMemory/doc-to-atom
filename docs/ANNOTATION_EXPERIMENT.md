# Paired reasoning-effort pilot and atomization release

```powershell
uv run scripts/run_annotation_experiment.py `
  --source data/squad_molab_raw.jsonl `
  --output-dir outputs/annotation-effort-pilot-20261030 `
  --pilot-documents 2 --workers 2
```

This is a paid API run. The output directory must not already exist. No automatic
paid retries occur. MiniMax-M3.1-Flash uses reasoning depths low, medium, high,
xhigh, and max on the **same first two training documents**. Validation documents
are not used to choose a depth. Each trial runs the entire three-stage pipeline;
atom banks and generated probes may therefore differ across depths.

The output cap is the documented MiniMax maximum, 524288 tokens per request,
including reasoning, not the old local cost budget of 8192. A 1M context window
is not a 1M output limit. Requests have a 30-minute timeout, no automatic retries,
and at most two simultaneous requests.

## Quality and selection

The annotation prompt explicitly requires pairwise-disjoint gold, supporting,
and distractor lists. Gold is minimal sufficient evidence, including indispensable
multi-hop bridge atoms. Supporting is optional evidence, not a second copy of
gold. Empty supporting/distractor lists are valid. A short XML example and a
final disjointness checklist reinforce the contract.

Raw provider overlaps are measured **before** deterministic canonicalization.
Otherwise, a canonicalized overlap count of zero could hide the original problem.

Only trials passing source-aware validation are eligible: complete source record
and question coverage, unchanged questions and answer variants, correct source
contexts/splits, no invalid source spans, grounded gold evidence according to
the dataset adapter, valid references and parse statuses. Selection then minimizes
raw overlaps, raw role parse notices, total tokens, and summed request latency.
The selected pilot records/responses are reused in production instead of paid
duplicate generation. Remaining train and validation records use the selected depth.

These are mechanical quality proxies, **not human-verified annotation accuracy**.
This pilot uses one sample per document/depth, no random seeds or repeated trials;
cache warming, shared server load, and two concurrent workers confound latency.
Token counts are not billed dollar costs (cache discounts and pricing differ).
Generation differences can also change probe difficulty and atom granularity.

## Artifacts

```text
<output-dir>/
  raw_source.jsonl             # byte-for-byte original input
  manifest.json               # model, settings, source/prompt hashes, pilot IDs
  prompts.json                # exact prompt text and dataset adapters
  comparison.json             # all five efforts, tokens, latency, source audits
  selection.json              # chosen depth and rule
  release.json                # training_ready gate, per-split audits
  benchmark/<effort>/
    atoms.jsonl
    raw_responses.jsonl        # full reasoning, usage, status, per-call latency
    raw_source.jsonl
    metrics.json
    validation.json
  production/
    atoms.jsonl
    raw_responses.jsonl
    raw_source.jsonl
    metrics.json
    validation.json
    train/                    # matching atoms, responses, source, audit
    validation/               # same paired artifacts for validation
```

Never train when `release.json` has `training_ready: false`. A failed release is
retained for inspection; it is not silently retried, patched, or promoted. This
workflow does not start training or overwrite older atomization results.

Independent source-aware validation commands:

```powershell
uv run scripts/validate_data.py `
  --input <output-dir>/production/train/atoms.jsonl `
  --source <output-dir>/production/train/raw_source.jsonl --reject-span-mismatch
uv run scripts/validate_data.py `
  --input <output-dir>/production/validation/atoms.jsonl `
  --source <output-dir>/production/validation/raw_source.jsonl --reject-span-mismatch
```

The validation CLI prints semantic counters; the experiment runner additionally
checks those counters and refuses a training-ready release if any gate fails.

The earlier one-record `atoms.jsonl` remains a historical baseline generated with
the old prompt at thinking max. Comparing it to the optimized max trial for the
same record can give a descriptive before/after overlap count, not a statistically
controlled prompt-quality estimate.

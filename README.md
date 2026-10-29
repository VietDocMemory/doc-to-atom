# Doc-to-Atom reproduction

Audit-first reproduction of **Doc-to-Atom: Learning to Compile and Compose Memory Atoms** (arXiv:2606.12400v1). This is an independent project. It does not implement “Doc-to-LoRA per chunk”.

The current audit and migration report is [docs/MINIMAX_MIGRATION_PAPER_AUDIT.md](docs/MINIMAX_MIGRATION_PAPER_AUDIT.md).
`PAPER_SPEC.md` records disclosed vs unspecified details; `REPRODUCTION_NOTES.md`
records assumptions. This is a **partial reconstruction**, not a verified numerical reproduction.

## Architecture

`document -> MiniMax XML atomization -> shared four-layer encoder -> atom compiler -> memory bank`

`query -> same encoder -> cosine K1=32 -> frozen BGE reranker -> top-K -> composer -> last-four-layer micro-LoRA -> frozen LLM`

Offline memory build and online query inference are separate APIs in `inference/build_memory.py` and `inference/query.py`.

## Commands

```bash
uv sync --extra test
uv run scripts/prepare_dataset.py --dataset squad --output data/squad_small_raw.jsonl
uv run scripts/atomize.py --input raw.jsonl --output atoms.jsonl
uv run scripts/validate_data.py --input atoms.jsonl
uv run scripts/smoke_train.py --config configs/resource_constrained.yaml
uv run scripts/train.py --config configs/resource_constrained.yaml --data atoms.jsonl
uv run scripts/evaluate.py --predictions outputs/predictions.jsonl
uv run pytest
```

## MiniMax-only annotation

Put `MINIMAX_API_KEY` in the ignored `.env` (see `.env.example`). Annotation uses
only this credential and the fixed official endpoint
`https://api.minimax.io/v1/chat/completions`; no separate base-URL environment
variable is needed. `HF_TOKEN` is optional for downloading gated base-model weights.
W&B credentials are only needed for optional online experiment tracking.

```bash
uv run doc2atom-atomize \
  --input data/squad_small_raw.jsonl \
  --output data/squad_small_atoms_m31.jsonl \
  --model MiniMax-M3.1-Flash --resume
```

Default model **MiniMax-M3.1-Flash differs from the paper's MiniMax-M2.5**;
new records explicitly mark this deviation. Prompts are reconstructed, not official.
`--model MiniMax-M2.5` can request the paper's annotator if it remains available to
your account; this does not make prompts, data splits or loss weights official.
`--provider minimax` remains accepted for old MiniMax commands; other providers are removed.

The default `--max-completion-tokens 8192` is a cost budget assumption, **including
thinking**, not a paper value or the vendor's recommended maximum. A `length` or
empty completion is saved to the sidecar and rejected; inspect it before increasing
the budget. M3.1-Flash uses provider-controlled temperature (1), not deterministic
temperature-zero decoding. No automatic paid retries occur. HTTP 402 means check
MiniMax billing, not reinstall Python packages.

Old datasets and raw sidecars remain readable with their original provenance;
cache reuse does not relabel old annotations as MiniMax. Use a **new output file**
for a clean M3.1 experiment. Explicit `--resume --retry-invalid` preserves the
entire previous JSONL as a content-addressed `.before-retry-*.jsonl` snapshot.

## Notebooks

- `notebook_molab_merged.py`: recommended button-gated local/Molab UI; commands
  run in the repository's uv environment. Open locally with
  `uvx --from marimo==0.25.2 marimo edit --no-sandbox notebook_molab_merged.py`.
- `notebook.py`: native Python variant, requiring dependencies in the kernel.
- The obsolete duplicate `notebook2.py` has been removed.

Start with Smoke, then validate before a longer run. These notebooks are single-GPU;
two 16-GiB GPUs do not become one 32-GiB GPU. No Molab run was performed for this migration.

See `docs/MOLAB.md` for the paper distributed profile and single-GPU fallback.
For the actual 2×16-GiB local / 1×96-GiB Molab setup, supported commands,
runtime limitations and the benchmark report form are in
`docs/LOCAL_MOLAB_RUNBOOK.md` and `docs/EVALUATION_SUMMARY_TEMPLATE.md`.
The 500-record local continuous shortened-curriculum experiment and its
measured-results reporting commands are documented in `docs/MEDIUM_LOCAL_RUNBOOK.md`.

W&B tracking is local-first. Set `runtime.wandb_mode: offline` (the default in the
Molab reconstruction and local diagnostic profiles) to avoid authentication and
network access during a run; artifacts are written under the run directory and can
be synchronized later. Use `disabled` only when no W&B run directory is wanted.

## Checkpoint retention

Training keeps at most two committed full `step-*` checkpoint directories after a
successful save: `best.json` points to the best configured early-stopping metric,
and `last.json` points to the most recent exact-resume state.
They may point to the same directory. A validation pass alone does not save: best
improvement, `training.checkpoint_every_steps`, or a terminal optimizer step does.
Use `doc2atom-train ... --resume latest` to resume from `last.json`. A hard crash
can lose steps after that last committed save, but resumed optimizer, scheduler,
RNG, curriculum, sampler and early-stopping state come from the full checkpoint.

`training.checkpoint_retention: best_last` and
`runtime.checkpoint_upload: best_weights` are the default policies; set the latter
to `none` to disable model artifact upload. W&B receives training metrics throughout
the run, but only the final best model weights, resolved config and run manifest
are uploaded as a model artifact after a normal/early-stopped completion. It does
not upload full resume state, ZeRO optimizer shards, or `.raw-responses`. If a Molab
instance disappears before completion, this policy provides no remote exact-resume
backup; copy the local `last` checkpoint to durable storage separately if needed.
An intentional mid-run stop saves `last` but defers the best artifact; a small
upload marker prevents re-uploading the same best step on a completed-run resume.

The upstream repository linked by the paper was audited at commit
`0848bda2aaef69b375436e24a30c8469a3f0560c`; it contained only a two-line
README, so no upstream implementation or checkpoint was available to reuse.


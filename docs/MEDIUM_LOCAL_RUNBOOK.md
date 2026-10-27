# Medium-scale local run: continuous shortened curriculum

Actual available hardware in this session: `nvidia-smi` reports **one RTX
4080, 16,376 MiB**; a second GPU is not visible. C: has ~461 GB free at the
time of setup. SQuAD extraction succeeded: `data/medium_squad_raw.jsonl`
contains **500 train + 100 validation distinct contexts** (600 JSONL lines,
622,008 bytes). This preparation selects the **first observed question per
context** to control annotation/API cost. This is a dataset-sampling deviation.
No project `.env` or annotator credentials were present. Consequently
atomization, Gemma training, OOM observations and F1/ROUGE-L are **not yet
available**. Never report them as passed/zero. The first atomization pass
requires network access and may incur external API costs (three calls per
document, plus retries).

## One continuous run, not four checkpoint-incompatible runs

`scripts/prepare_sanity_configs.py` generates **independent** starts at the
paper steps 0/2000/6000/9000. It cannot warm-start weights or optimizer across
different config fingerprints. For this medium experiment use
`scripts/prepare_medium_config.py`: it creates **one** config with optional
shortened boundaries 200/400/600 and `max_steps=800`. A single run and one
optimizer pass stage 1 → 2 → 3 → 4 without resetting parameters, unlike four
separate `scripts/train.py` commands. Defaults remain the paper boundaries
when `curriculum_boundaries` is absent. Shortening them is marked as a
reproduction deviation, **not** a paper-faithful training result.

Its actual batch is **one document per micro-step**, 16 accumulation steps;
gradient accumulation is not a per-device batch of 16. With 500 train records
and 800 optimizer steps it may need roughly 25–28 epochs, not just four short
passes. Early stopping/best selection is delayed until stage 4 so that
`best.json` points to a stage-4 checkpoint; the experimental patience is
recorded in the config. No guarantees of memory fit or loss convergence exist
before executing on the real hardware. Both checkpoints are retained (`best`
and `last`), not top-3 best. Large raw response logs can increase disk use.

## PowerShell from repository root

```powershell
uv sync --extra train --extra test
uv run python -m pytest tests/ -q
# Already produced in this workspace. On a clean machine run once:
if (!(Test-Path data/medium_squad_raw.jsonl)) {
  uv run python scripts/prepare_dataset.py --dataset squad --train-records 500 --validation-records 100 --output data/medium_squad_raw.jsonl
}
# Put MINIMAX_API_KEY and MINIMAX_BASE_URL in an ignored .env; do not commit secrets.
# Never overwrite partial atomization: --resume continues existing output.
if (Test-Path data/medium_squad_atoms.jsonl) {
  uv run python scripts/atomize.py --input data/medium_squad_raw.jsonl --output data/medium_squad_atoms.jsonl --resume
} else {
  uv run python scripts/atomize.py --input data/medium_squad_raw.jsonl --output data/medium_squad_atoms.jsonl
}
uv run python scripts/validate_data.py --input data/medium_squad_atoms.jsonl --source data/medium_squad_raw.jsonl --reject-span-mismatch
```

Check the validation JSON for `errors=0` and `invalid_spans=0` before
continuing. Atom-only records from failed annotation do **not** produce
training examples; count their failures and rerun or explicitly document data
loss. The train CLI validates spans, but evaluation also validates the source
contract. `--resume` does not automatically retry cached records with bad
spans: use `--retry-invalid` only after reviewing that intended behavior.

```powershell
# Already produced: outputs/local/medium_run/configs/medium.yaml
if (!(Test-Path outputs/local/medium_run/configs/medium.yaml)) {
  uv run python scripts/prepare_medium_config.py --base configs/reconstruction/molab_single_gpu.yaml --output outputs/local/medium_run/configs/medium.yaml --train-records 500 --stage-steps 200
}
$env:CUDA_VISIBLE_DEVICES="0"    # nvidia-smi currently lists one RTX 4080
# Optional separate 1-step diagnostic; do not resume it into the 800-step run.
uv run python scripts/train.py --config outputs/local/medium_run/configs/medium.yaml --data data/medium_squad_atoms.jsonl --output outputs/local/medium_preflight --max-steps 1 --wandb-mode disabled
# Only after confirming a real 16-GiB model step fits, launch the full run.
uv run python scripts/train.py --config outputs/local/medium_run/configs/medium.yaml --data data/medium_squad_atoms.jsonl --output outputs/local/medium_run --wandb-mode offline
```

Monitor `nvidia-smi`, free disk (`Get-PSDrive C`), and
`outputs/local/medium_run/train_log.jsonl`. Check `curriculum_stage`, `top_k`,
the six loss fields, `train.generated_irrelevant.no_memory_weight` and
`train.generated_irrelevant.adapter_norm`; absence of generated irrelevant
questions should be reported, not inferred as a successful refusal test.
Stop on CUDA OOM or nonfinite loss, and record the step. Resume only the **same
800-step config and same data** with `--resume latest` and the same output.
There is no automatic OOM-recovery/cuda-memory throttling. If 16 GiB does not
fit even at context 1024, reducing the context further is a new explicit
deviation; do not pretend that switching to `--nproc_per_node=2` shares VRAM.

## Validation evaluation: stage-4 best

```powershell
uv run python scripts/evaluate.py --config outputs/local/medium_run/config/resolved.yaml --data data/medium_squad_atoms.jsonl --source data/medium_squad_raw.jsonl --checkpoint outputs/local/medium_run/checkpoints/best.json --split validation --output outputs/local/medium_run/predictions/val.jsonl --wandb-mode disabled
uv run python scripts/report_medium_run.py --run outputs/local/medium_run --output outputs/local/medium_run/reports/medium_summary.json
```

The evaluator creates `metrics/evaluation_validation.json` with F1, ROUGE-L,
refusal F1, per-question latency and evaluation peak VRAM. The report script
refuses to invent unobserved metrics and checks that all four stages occur in
the same training log. Evaluation also needs audited *source answers*: any
rescued generated probe without a source answer must be excluded or provided
a verified answer under a documented policy before score generation.

| Preliminary measured metric | Result in this workspace |
|---|---|
| Source JSONL train / validation | 500 / 100 records |
| Atomized valid records | Not measured — annotator credentials absent |
| Stage 1/2/3/4 optimizer steps and no-OOM | Not measured — training not started |
| Validation F1 / ROUGE-L | Not measured |
| Refusal F1 / mean latency | Not measured |

These values **cannot** be compared directly with the full-dataset paper
tables; 600 sampled records and shortened stages are local diagnostics.

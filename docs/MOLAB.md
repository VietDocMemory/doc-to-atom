# Molab setup and execution

## Exact distributed profile

```bash
uv sync --extra train --extra test
accelerate launch --config_file configs/accelerate_paper.yaml \
  scripts/train.py --config configs/gemma_paper.yaml --data DATA.jsonl
```

Use 8 x A100 80GB, bf16, FlashAttention-2 and `find_unused_parameters=true` to match the paper runtime. Training cannot start from the paper config until unpublished loss coefficients and `rho` are supplied explicitly.

## Single-GPU Molab fallback

```bash
uv sync --extra test
uv run scripts/smoke_train.py --config configs/resource_constrained.yaml
uv run scripts/train.py --config configs/resource_constrained.yaml --data DATA.jsonl
```

Every fallback is written to `outputs/run_manifest.json`. Do not copy resource-constrained assumptions into `gemma_paper.yaml`.

Parameter accounting and the limits of the GPU estimate are recorded in
`docs/RESOURCE_ESTIMATE.md`.


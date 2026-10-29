# Local → Molab runbook (actual hardware)

This is a **reconstruction**, not the paper's 8×A100 run. Model weights, benchmark
JSONL and annotation credentials are not bundled. Preserve separate outputs and
`config/resolved.yaml` for every run. See `EVALUATION_SUMMARY_TEMPLATE.md`.

## Supported capabilities and hard limits

- Use `scripts/train.py` or `python -m doc_to_atom.training.cli`, **not**
  `python -m doc_to_atom.training.loop`. There is no `--override` argument.
- The training loop processes **one document per micro-step**, without a
  DataLoader. Declaring per-device batch 2/8/16 or workers/prefetch/persistence
  does not implement them. The four Molab CPU cores run the one-process loop;
  do not claim `num_workers=2` until a real collated DataLoader is added.
- The training CLI uses `LocalTrainingRuntime`; `torchrun --nproc_per_node=2
  scripts/train.py` launches **two independent runs, not DDP**. Existing
  `AccelerateTrainingRuntime` is exercised by integration workers, not wired
  into this CLI. On native Windows, the Linux NCCL/DeepSpeed profile is also
  unsuitable. Both local GPUs can run *independent* stage diagnostics.
- `configs/reconstruction/molab_zero2.yaml` is an eight-rank recipe:
  micro=1, accumulation=16, global=128; its Accelerate file requests 8 ranks.
  On one GPU use `configs/reconstruction/molab_blackwell_1gpu.yaml`:
  micro=1, accumulation=16, global=16, single-process AdamW. Simply setting
  the backend name to ZeRO-2 would not turn it on. Base weights are BF16 and
  request FlashAttention-2 (factory warns about SDPA fallback). Gradient
  checkpointing is disabled for the hook-scoped adapter.
- Checkpoint retention is **best + last**, at most TWO full checkpoints, not
  top-three best. Set `checkpoint_every_steps` in a separate documented config
  if needed, monitor free space, and do not delete `last` before a resume.
  `train_log.jsonl` contains per-question traces. W&B `offline` writes locally,
  `disabled` avoids W&B directories; TensorBoard is not implemented.

## 1. Local preflight and SQuAD sample — PowerShell

```powershell
uv sync --extra train --extra test
uv run python -m pytest tests/ -q
uv run python scripts/prepare_dataset.py --dataset squad --train-rows 16 --validation-rows 8 --output data/squad_sanity_raw.jsonl
# Load only MINIMAX_API_KEY from the ignored .env for annotation.
uv run python scripts/atomize.py --input data/squad_sanity_raw.jsonl --output data/squad_sanity_atoms.jsonl
uv run python scripts/validate_data.py --input data/squad_sanity_atoms.jsonl
uv run python scripts/train.py --config configs/reconstruction/molab_single_gpu.yaml --data data/squad_sanity_atoms.jsonl --validate-only --output outputs/local/preflight
uv run python scripts/prepare_sanity_configs.py --base configs/reconstruction/molab_single_gpu.yaml --output outputs/local/stage-configs --steps 2
```

In **PowerShell terminal A** run GPU 0, stages 1/2:

```powershell
$env:CUDA_VISIBLE_DEVICES="0"
foreach ($stage in @("stage1", "stage2")) {
  uv run python scripts/train.py --config "outputs/local/stage-configs/$stage.yaml" --data data/squad_sanity_atoms.jsonl --output "outputs/local/$stage" --wandb-mode disabled
}
```

In **terminal B**, set `$env:CUDA_VISIBLE_DEVICES="1"` and change the list to
`@("stage3", "stage4")`. Each terminal loads its own Gemma and CPU BGE; these
diagnostics are not synchronous DDP. Monitor `nvidia-smi`; stop if either
16-GiB GPU OOMs. The generated configs shorten context to 1024 and answers to
64 tokens. They start independently at offsets 0/2000/6000/9000 (K=1/4/8/8)
for two optimizer steps each, **not** a continuous curriculum or evidence of
long-run no-OOM/no-drift. Inspect `train_log.jsonl` for finite losses,
`curriculum_step`, `train.generated_irrelevant.adapter_norm` and group means
`train.*.no_memory_weight`. A randomly initialized two-step run need
not refuse correctly; the controlled null-candidate regression is in
`tests/test_router_composer.py`.

## 1a. Same supported flow — Bash/Linux

```bash
uv sync --extra train --extra test
uv run python -m pytest tests/ -q
uv run python scripts/prepare_dataset.py --dataset squad --train-rows 16 --validation-rows 8 --output data/squad_sanity_raw.jsonl
uv run python scripts/atomize.py --input data/squad_sanity_raw.jsonl --output data/squad_sanity_atoms.jsonl
uv run python scripts/prepare_sanity_configs.py --base configs/reconstruction/molab_single_gpu.yaml --output outputs/local/stage-configs --steps 2
for stage in stage1 stage2; do
  CUDA_VISIBLE_DEVICES=0 uv run python scripts/train.py --config "outputs/local/stage-configs/$stage.yaml" --data data/squad_sanity_atoms.jsonl --output "outputs/local/$stage" --wandb-mode disabled
done
for stage in stage3 stage4; do
  CUDA_VISIBLE_DEVICES=1 uv run python scripts/train.py --config "outputs/local/stage-configs/$stage.yaml" --data data/squad_sanity_atoms.jsonl --output "outputs/local/$stage" --wandb-mode disabled
done
```

Run loops in separate shells for simultaneous GPU use. Place `HF_HOME`,
`HF_DATASETS_CACHE` and (if offline) `WANDB_DIR` on a volume with enough free
space. Check `Get-PSDrive` / `df -h` periodically; the repository has no byte
quota. Do not remove unrecognized files or other runs to free space.

## 2. Molab: actual one-GPU full-dataset run — Bash

Create audited atomized schema-v2 JSONL for the five in-domain datasets with
document-disjoint train/validation/test splits. `prepare_dataset.py` prepares
only a small SQuAD subset; `data/preprocessors.py` provides dataset mappings,
not downloads for all benchmarks. Obtain permitted source records, atomize
them, validate source spans and keep all eight LongBench subsets **held out**.

```bash
uv sync --extra train --extra test
uv run python -m pytest tests/ -q
# Do NOT use --run-molab on one GPU: the marked test requires 2 GPUs/NCCL.
uv run python scripts/train.py --config configs/reconstruction/molab_blackwell_1gpu.yaml --data data/in_domain_train_val_atoms.jsonl --validate-only --output outputs/molab/validation
uv run python scripts/train.py --config configs/reconstruction/molab_blackwell_1gpu.yaml --data data/in_domain_train_val_atoms.jsonl --output outputs/molab/gemma_full --wandb-mode offline
```

Resume with the **same config, data and output** plus `--resume latest`.
`max_epochs=50` and early stopping are inherited. Monitor `nvidia-smi`,
`outputs/molab/gemma_full/train_log.jsonl` and
`reports/training_result.json` for peak allocated VRAM and losses. W&B
`--wandb-mode online` is optional and sends telemetry to the configured
account. Effective batch 16 is a deviation from paper batch 128. Batch 8/16
documents per GPU and single-GPU ZeRO-2 need implementation and verification
before using those labels in a run report.

## 3. Evaluate each test dataset (Bash)

```bash
uv run python scripts/evaluate.py --config outputs/molab/gemma_full/config/resolved.yaml --data data/squad_test_atoms.jsonl --source data/squad_test_raw.jsonl --checkpoint outputs/molab/gemma_full/checkpoints/best.json --split test --output outputs/molab/gemma_full/squad/predictions/test.jsonl --wandb-mode offline
```

Repeat with matching raw and atomized `test` files for DROP, ROPES,
2WikiMultiHopQA, QASPER and all eight held-out LongBench subsets; change the
`squad` output directory to a **unique** dataset directory each time so that
`metrics/evaluation_test.json` cannot be overwritten. The evaluator requires
source JSONL, matching question IDs and checkpoint/config fingerprint.
Answerable rescued probes without verified source answers must be excluded
from evaluation rather than scored against invented text. Its `refusal_f1` is
binary F1 on both answerable and irrelevant classes: check sample composition
before comparing it with the paper's refusal numbers. Score-only mode is
`uv run python scripts/evaluate.py --predictions PATH`.

Paper Table 3 reports *mean per-update compile latency and additional compile
GPU memory*, not the full-training `elapsed_seconds` or peak allocated VRAM
already logged here. Those quantities require a separate matching measurement
protocol; do not label training peak as Table 3 memory.

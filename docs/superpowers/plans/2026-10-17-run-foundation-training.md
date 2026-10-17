# Run Foundation and Training Reliability Implementation Plan
> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every training run reproducible, resumable at optimizer boundaries, observable through W&B (with offline fallback), and safe to launch on Molab without changing the research architecture.

**Architecture:** Add a typed run context and versioned checkpoint state around the existing custom loop. Preserve `DocToAtomSystem` and the multi-objective implementation; make step semantics, artifact layout, and configuration provenance explicit. Checkpoints are atomic directories containing model and trainer state plus hashes, rather than an ambiguous single weights file.

**Tech Stack:** Python 3.12, PyTorch, Transformers scheduler, Accelerate/DeepSpeed on Linux, Pydantic/YAML, Weights & Biases, pytest.

**Spec:** [Full Pipeline Readiness Design](../specs/2026-10-17-full-pipeline-readiness-design.md)

## Global Constraints

- Do not run full training; use synthetic/tiny fixtures only.
- Preserve paper-like routing and keep Micro-KV disabled.
- Fail fast if gradient checkpointing is requested until recomputation equivalence is proven.
- Never log credentials or `.env` content.
- A checkpoint is valid only after its completion marker is atomically published.
- Resume only from optimizer boundaries; reject incompatible config/model/data hashes unless an explicit unsafe override is recorded.

## Review Focus

- Exact resume equivalence, off-by-one errors in data cursor and scheduler, distributed-safe writes, and artifact provenance.
- No silent architecture or effective-batch changes.

---

## Task 1: Normalize configuration families and validate invariants

**Files:**
- Create: `configs/paper_like/base.yaml`
- Create: `configs/reconstruction/molab_zero2.yaml`
- Create: `configs/experiments/small_overfit.yaml`
- Create: `configs/runtime/local_offline.yaml`
- Modify: `src/doc_to_atom/config.py`
- Test: `tests/test_config.py`

- [x] Write failing tests for recursive base resolution, cycle detection, missing disclosed values, config family metadata, and stable config hashing.
- [x] Add `load_and_resolve_config`, schema/invariant validation, canonical serialization, and SHA-256 helpers while retaining `load_config` compatibility.
- [x] Encode every unpublished value under `assumptions`; encode every deviation under `deviations`.
- [x] Add Molab ZeRO-2 reconstruction config with gradient checkpointing and Micro-KV explicitly false.
- [x] Run `py -m uv run pytest tests/test_config.py -q`.
- [x] Commit: `feat(config): separate paper assumptions experiments and runtime`.

## Task 2: Introduce the run artifact and tracking contract

**Files:**
- Create: `src/doc_to_atom/runtime/__init__.py`
- Create: `src/doc_to_atom/runtime/run.py`
- Create: `src/doc_to_atom/runtime/tracking.py`
- Modify: `pyproject.toml`
- Test: `tests/test_run_context.py`

- [x] Write failing tests for deterministic manifest fields, output directory layout, redaction, W&B disabled/offline modes, and stable run IDs on resume.
- [x] Implement `RunContext.create(...)`, atomic JSON/YAML writes, environment/package/git capture, and structured JSONL events.
- [x] Implement `RunTracker` with `online -> offline` fallback and a no-op backend; log config, manifest, metrics, files, and checkpoint artifacts through one interface.
- [x] Add bounded `wandb` and `safetensors` dependencies and document network-free initialization.
- [x] Run `py -m uv run pytest tests/test_run_context.py -q`.
- [x] Commit: `feat(runtime): add reproducible run and wandb artifact contract`.

## Task 3: Replace checkpoint v1 with atomic checkpoint v2

**Files:**
- Modify: `src/doc_to_atom/training/checkpoint.py`
- Create: `src/doc_to_atom/training/state.py`
- Modify: `src/doc_to_atom/training/smoke.py`
- Test: `tests/test_losses_curriculum_checkpoint.py`
- Test: `tests/test_resume_equivalence.py`

- [x] Write failing round-trip tests for trainable modules, optimizer, scheduler, scaler, epoch, record cursor, micro/optimizer/curriculum steps, early-stop state, Python/NumPy/CPU/CUDA RNG, sampler state, hashes, and W&B run ID.
- [x] Define a versioned `TrainerState`; capture/restore all RNG sources and validate checkpoint compatibility.
- [x] Save into a temporary sibling directory, fsync metadata where supported, then atomically rename and write `COMMITTED`.
- [x] Retain an explicit legacy loader for current `.pt` smoke checkpoints; never silently treat legacy state as fully resumable.
- [x] Add interrupted-vs-uninterrupted deterministic equivalence test on a tiny CPU model.
- [x] Run `py -m uv run pytest tests/test_losses_curriculum_checkpoint.py tests/test_resume_equivalence.py -q`.
- [x] Commit: `feat(training): add complete atomic resumable checkpoints`.

## Task 4: Correct training step semantics and early stopping

**Files:**
- Modify: `src/doc_to_atom/training/loop.py`
- Modify: `src/doc_to_atom/training/curriculum.py`
- Modify: `src/doc_to_atom/training/trainer.py`
- Test: `tests/test_training_loop.py`
- Test: `tests/test_gradient_flow.py`
- Test: `tests/test_one_example_overfit.py`

- [x] Write failing tests showing scheduler and curriculum advance only on `optimizer_step`, partial accumulation flush is handled exactly once, and resume begins at the next unconsumed record.
- [x] Refactor into explicit micro-step and optimizer-step boundaries; persist deterministic epoch ordering and cursor.
- [x] Add configurable early stopping with metric, mode, patience, minimum delta, warmup validations, and best-checkpoint pointer.
- [x] Log objective components, gradient norm, LR, curriculum stage, timing, memory, and validation state at optimizer boundaries.
- [x] Add gradient-flow assertions for encoder projection, compiler, router, and frozen base model.
- [x] Add a bounded one-example overfit test marked `integration`.
- [x] Run `py -m uv run pytest tests/test_training_loop.py tests/test_gradient_flow.py tests/test_one_example_overfit.py -q`.
- [x] Commit: `fix(training): make optimizer step resume and stopping exact`.

## Task 5: Complete the training CLI and Molab launch path

**Files:**
- Modify: `src/doc_to_atom/training/cli.py`
- Modify: `src/doc_to_atom/training/factory.py`
- Create: `scripts/molab/train_zero2.sh`
- Create: `configs/accelerate/molab_zero2.yaml`
- Modify: `README.md`
- Test: `tests/test_training_cli.py`

- [x] Write CLI tests for fresh run, `--resume PATH`, `--resume latest`, incompatible checkpoint rejection, `--wandb-mode offline`, and config snapshot creation.
- [x] Wire `RunContext`, tracker, checkpoint manager, and resolved config into the CLI.
- [x] Turn gradient checkpointing into an actionable fail-fast error that explains the ContextVar/recomputation limitation; retain Micro-KV `NotImplementedError` with documentation pointer.
- [ ] Add Linux/Molab Accelerate + DeepSpeed launch script that validates paths, GPU visibility, writable output, and offline W&B directories before starting.
- [x] Run `py -m uv run pytest tests/test_training_cli.py -q` and `py -m uv run pytest -q`.
- [ ] Commit: `feat(training): expose fresh and resumable Molab CLI`.

## Acceptance Commands

```powershell
py -m uv sync --extra train --extra test
py -m uv run pytest -q -m "not gpu and not molab"
py -m uv run doc2atom-train --config configs/experiments/small_overfit.yaml --output outputs/ci-train --wandb-mode offline
py -m uv run doc2atom-train --config configs/experiments/small_overfit.yaml --output outputs/ci-train --resume latest --wandb-mode offline
```


# Accelerate and DeepSpeed ZeRO-2 Correctness Implementation Plan
> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make single-GPU and multi-GPU training share exact optimizer-step, curriculum, checkpoint, and resume semantics, with Accelerate owning distributed execution and DeepSpeed ZeRO-2 owning sharded optimizer state.

**Architecture:** Introduce a narrow distributed runtime around Accelerate. The training system exposes one dispatching `forward` so DDP/DeepSpeed wrappers observe every differentiable path. A deterministic rank sampler pads complete accumulation windows equally across ranks. Local checkpoint v2 remains the portable/evaluation artifact; distributed checkpoints additionally contain `Accelerator.save_state` output and publish only after every rank succeeds.

**Tech Stack:** PyTorch distributed/Gloo/NCCL, Hugging Face Accelerate 1.x, DeepSpeed ZeRO-2, safetensors, pytest, torchrun.

**Spec:** [Full Pipeline Readiness Design](../specs/2026-10-17-full-pipeline-readiness-design.md)

## Global Constraints

- No full training and no tuning.
- Evaluation remains blocked until the Molab 2-GPU gate passes.
- No rank may independently choose optimizer/global/curriculum step.
- Checkpoint pointers may reference only directories carrying `COMMITTED` after all ranks finish.
- Any unsupported topology, uneven accumulation contract, stale world size, or partial checkpoint fails before training.

## Review Focus

- Duplicate or dropped samples, wrong loss scaling, double scheduler steps, rank-divergent curriculum, rank-0-only optimizer state, premature checkpoint publication, and resume into a different topology.

---

## Task 1: Define distributed sampler and rank-invariant counters

**Files:** Create `src/doc_to_atom/training/distributed.py`; create `tests/test_distributed_sampler.py`.

- [x] RED tests for deterministic epoch permutations, equal rank lengths, full accumulation windows, explicit padding inventory, disjoint non-padding samples, cursor resume, and topology mismatch.
- [x] Implement `DistributedEpochPlan` from seed/epoch/world-size/accumulation and serialize its contract for `TrainerState.sampler_state`.
- [x] Verify `py -m uv run pytest tests/test_distributed_sampler.py -q`.
- [x] Commit `feat(training): add deterministic distributed epoch plans`.

## Task 2: Put every trainable forward behind the wrapped model

**Files:** Modify `src/doc_to_atom/training/system.py`, `src/doc_to_atom/training/loop.py`; modify `tests/test_injection_system.py`.

- [x] RED tests for build-memory, teacher, student, and forced-gold dispatch through `TeacherStudentSystem.forward`.
- [x] Refactor loop calls so the prepared model wrapper observes all trainable computation; use direct module access only for state and mode changes.
- [x] Verify local single-GPU behavior remains equivalent.
- [x] Commit `refactor(training): route trainable work through wrapped system`.

## Task 3: Give Accelerate optimizer and scheduler ownership

**Files:** Modify `src/doc_to_atom/training/distributed.py`, `src/doc_to_atom/training/loop.py`, `src/doc_to_atom/training/cli.py`, `src/doc_to_atom/training/factory.py`; create `tests/test_distributed_runtime.py`.

- [x] RED contract tests proving prepare happens once, backward/clip/step happen on all ranks, logging happens on rank 0, and one optimizer update advances scheduler/curriculum exactly once.
- [x] Use Accelerate accumulation semantics; configure deterministic padding so no partial window is required in distributed mode.
- [x] Validate effective batch as microbatch × accumulation × world size.
- [x] Keep the user-facing distributed fail-fast until the Task 5 and Molab Task 6 gates pass.
- [x] Commit `feat(training): integrate accelerate-owned optimization`.

## Task 4: Publish and restore atomic distributed checkpoints

**Files:** Modify `src/doc_to_atom/training/checkpoint.py`, `src/doc_to_atom/training/distributed.py`, `src/doc_to_atom/training/state.py`; create `tests/test_distributed_checkpoint.py`.

- [x] RED tests for all-rank save participation, rank-0 publication, failure before `COMMITTED`, pointer ordering, world-size/backend mismatch, Accelerate load before cursor use, and portable weights.
- [x] Write to a deterministic incomplete sibling, call `Accelerator.save_state` on every rank, barrier, write portable checkpoint/metadata on rank 0, barrier, then atomically publish.
- [x] Restore model/optimizer/scheduler/scaler/RNG through `Accelerator.load_state`; restore control state only after compatibility validation.
- [x] Commit `feat(training): add atomic distributed checkpoint resume`.

## Task 5: Add 1-process versus 2-process regression harness

**Files:** Create `tests/distributed_worker.py`, `tests/test_distributed_equivalence.py`; modify `pyproject.toml`.

- [x] Run a tiny deterministic model through single-process, two-process Gloo, and interrupted two-process resume.
- [x] Compare optimizer step, scheduler LR, curriculum step, consumed global sample IDs, portable trainable weights, optimizer tensors, and next RNG draws at `atol=1e-7`, `rtol=1e-6` (counters/order exact).
- [x] Make subprocess timeout, rank crash, and partial artifact diagnostics explicit.
- [x] Commit `test(training): prove two-process distributed resume equivalence`.

## Task 6: Enable and verify Molab ZeRO-2

**Files:** Modify `scripts/molab/train_zero2.sh`, `configs/accelerate/molab_zero2.yaml`, `configs/deepspeed_zero2.json`; create `tests/test_molab_zero2.py`; modify `PRE_FULL_RUN_CHECKLIST.md`.

- [x] Prepare and unit-test preflight for CUDA/NCCL/DeepSpeed versions, GPU count, writable storage, effective batch, and config/checkpoint hashes.
- [ ] Run `pytest -m molab` on two Molab GPUs using a tiny fixture, then run fresh/interrupted/resume equivalence.
- [ ] Save logs and reports under the run directory and W&B offline.
- [ ] Remove the launcher fail-fast only after the 2-GPU evidence artifact is present and validated.
- [ ] Commit `feat(molab): enable verified deepspeed zero2 training`.

## Acceptance Gates

```powershell
py -m uv run pytest tests/test_distributed_sampler.py tests/test_distributed_runtime.py tests/test_distributed_checkpoint.py -q
py -m uv run pytest tests/test_distributed_equivalence.py -q
py -m uv run pytest -q -m "not molab and not network"
```

Molab only:

```bash
uv run pytest tests/test_molab_zero2.py -q -m molab --run-molab
DOC2ATOM_DATA=data/fixtures/distributed_atoms.jsonl DOC2ATOM_OUTPUT=outputs/molab-zero2-smoke scripts/molab/train_zero2.sh
```


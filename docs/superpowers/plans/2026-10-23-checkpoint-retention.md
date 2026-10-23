# Checkpoint Retention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bound local full checkpoints to best and last while preserving exact resume and uploading only best weights to W&B.

**Architecture:** Keep immutable v2 `step-*` directories and atomic pointers. A focused checkpoint-retention helper validates and prunes after safe publication; the training loop owns save triggers and rank barriers; tracking owns the model-only artifact.

**Tech Stack:** Python 3.12+, PyTorch, Accelerate/DeepSpeed ZeRO-2, W&B, pytest.

**Spec:** `docs/superpowers/specs/2026-10-23-checkpoint-retention-design.md`

## Global Constraints

- No changes to model architecture, losses, optimizer, or curriculum.
- Full `best` and `last` stay exact-resume capable; no deletion of optimizer, scheduler, RNG, sampler or trainer state.
- No upload of `.raw-responses` or every full checkpoint.
- Rank 0 alone mutates pointers and prunes; all ranks synchronize before and after cleanup.
- Preserve unrelated worktree changes (`notebook.py` and report deletions).

## Review Focus

- Malformed or escaping pointer: pruning must refuse to run (Task 1 test).
- Save failure or missing committed file: prior checkpoint must remain (Task 1 test).
- Best and last reference the same step: keep one full directory (Task 1 test).
- Validation occurs without improvement between intervals: no save (Task 2 test).
- ZeRO-2 portable weights differ in path from local weights: artifact selects the correct file and no optimizer shard (Task 3 test).

---

### Task 1: Safe retention primitive and config policy

**Files:** `src/doc_to_atom/training/checkpoint.py`, `src/doc_to_atom/config.py`, `configs/reconstruction/molab_zero2.yaml`, `tests/test_checkpoint_retention.py`, `tests/test_config.py` (or nearest existing config tests).

**Interfaces:** `verify_committed_checkpoint(path: Path, *, optimizer_step: int, distributed: bool) -> None`; `prune_unreferenced_checkpoints(checkpoint_dir: Path) -> list[Path]`. Config `training.checkpoint_retention: best_last`, `runtime.checkpoint_upload: best_weights|none`; absent keys use those defaults.

- [x] Write tests for complete local/distributed checkpoint validation, safe two-pointer pruning, shared target, malformed/outside pointer, missing/partial target and preservation of unrelated directories.
- [x] Run focused tests and observe expected failures.
- [x] Implement the two helpers and config validation with explicit `best_last`/`best_weights|none` policy.
- [x] Run focused tests green.

### Task 2: Save scheduling and exact-resume behavior

**Files:** `src/doc_to_atom/training/loop.py`, `tests/test_training_loop.py`, `tests/test_resume_equivalence.py`, `tests/test_distributed_checkpoint.py`.

**Interfaces:** Consume Task 1 helpers. `save_state(epoch, cursor, *, best=False)` preserves full v2 state; scheduler saves on improvement, positive `checkpoint_every_steps`, or terminal optimizer step. Reuse an already committed same-step checkpoint rather than overwriting it.

- [x] Add failing tests for non-improving validation, periodic `last`, best/last divergence and cleanup, early-stop final state, intentional interruption and resume equivalence, and rank-0-only cleanup with barriers.
- [x] Run focused tests red.
- [x] Change save triggers and publication order: save, verify, publish `last`/`best`, prune, synchronize. Terminal save occurs even if an earlier interval checkpoint exists.
- [x] Run focused tests green.

### Task 3: W&B best-weights-only artifact

**Files:** `src/doc_to_atom/runtime/tracking.py`, `src/doc_to_atom/training/loop.py`, `tests/test_run_context.py`, `tests/test_training_loop.py`.

**Interfaces:** `RunTracker.log_best_weights(checkpoint: Path, *, optimizer_step: int) -> None` selects local or `portable/model.safetensors`; `runtime.checkpoint_upload=none` skips it. Invoke once after the loop, not at each improved validation. Existing `log_metrics`, `log_config`, and `log_manifest` remain unchanged.

- [x] Add failing tests inspecting actual logged artifact/files: model weights only, no `training.pt`, `accelerate`, RNG or `.raw-responses`; upload only once for final best and disabled policy.
- [x] Run focused tests red.
- [x] Implement the model-only artifact and call it once after a completed run with a committed best.
- [x] Run focused tests green.

### Task 4: End-to-end verification and guidance

**Files:** relevant training guide/README and tests if a newly discovered gap needs a regression.

- [x] Document `best.json`/`last.json`, interval and W&B policy, crash recovery boundary, and old-run compatibility.
- [x] Run `uv run --extra test pytest` and check all failures by name.
- [x] Run bounded distributed checkpoint/resume tests on available hardware; report any ZeRO-2 hardware gate not exercised.
- [x] Review diff and status, confirming unrelated worktree changes are untouched.

# Local Checkpoint Artifact Logging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep two full local checkpoints and make W&B publish best model weights plus one terminal, exact-resume last checkpoint under explicit policy, without uploading anything during local implementation.

**Architecture:** Training remains local-first. A new upload policy uses `RunTracker` to create aliased best/last artifacts after committed-checkpoint verification; upload status is recorded separately from checkpoint retention and failure never mutates pointers.

**Tech Stack:** Python 3.12+, PyTorch/Accelerate/DeepSpeed checkpoint formats, W&B artifact API (fake/offline clients in tests), pytest.

**Spec:** `docs/superpowers/specs/2026-10-25-inference-evaluation-protocol-design.md`

## Global Constraints

- No live W&B upload, Molab run, GitHub push, API call or full training in this task.
- Best artifact contains model weights/config/manifest only; last contains all exact-resume checkpoint files; no raw provider responses.
- Existing `none` and `best_weights` configs/checkpoints retain behavior and fingerprints. New policy is opt-in for a future new run.
- Upload failure must not remove or invalidate locally committed best/last checkpoints.

## Review Focus

- W&B asynchronous return must not be labeled remotely durable; Task 1 tests wait/offline status.
- A failed upload leaves best/last pointers and full optimizer/RNG files intact; Task 2 tests it.
- ZeRO-2 last must include the distributed resume state as well as portable weights; Task 1 tests it.
- `raw-responses` must never enter either artifact; Task 1 tests it.
- Resumed terminal run must not enqueue the same last checkpoint twice; Task 2 tests marker matching.

---

### Task 1: Last full artifact and upload status

**Files:** Modify `src/doc_to_atom/runtime/tracking.py`; test `tests/test_run_context.py`.

**Interfaces:** `RunTracker.log_last_checkpoint(checkpoint: str | Path, *, optimizer_step: int) -> dict` returns `status` (`uploaded` online after wait, `offline_pending` offline, `disabled` without run), checkpoint basename, step and artifact name. `log_best_weights(...)` returns the same status convention while retaining its model-only artifact contents. The last artifact has alias `last`, includes checkpoint files at stable relative paths plus config and manifest, and excludes any `*.raw-responses*`.

- [ ] Write tests using a fake W&B artifact/run for single-GPU and ZeRO-2 directories, online wait for both best/last, offline pending, disabled mode, and exclusion of raw responses; assert exact-resume files are present.
- [ ] Run `uv run --extra test pytest tests/test_run_context.py -q` and observe RED.
- [ ] Implement the artifact method and status returns without changing existing best artifact contents.
- [ ] Re-run targeted tests to green.

### Task 2: Terminal policy and idempotent publication

**Files:** Modify `src/doc_to_atom/config.py`, `src/doc_to_atom/training/loop.py`; test `tests/test_config.py`, `tests/test_training_loop.py`.

**Interfaces:** `runtime.checkpoint_upload` accepts `none`, `best_weights`, `best_weights_and_last_full`; only the new option uploads last full at terminal/early-stop. `manifests/{best_model_upload,last_checkpoint_upload}.json` record checkpoint, optimizer_step and status; only `uploaded` is remote-durable. Existing two-field best markers are read compatibly but do not claim remote durability.

- [ ] Write tests for opt-in policy validation, terminal upload once, early-stop upload, resumed idempotency, local pointer/load integrity after upload failure, and no upload at nonterminal validation steps.
- [ ] Run targeted tests and observe RED.
- [ ] Implement policy branches after safe checkpoint commit and pointer update. Catch/report W&B upload failures as structured warnings; never disguise offline pending as uploaded.
- [ ] Re-run targeted tests to green.

### Task 3: Local verification and operator guide

**Files:** Update `FULL_TRAINING_GUIDE.md` (or existing checkpoint section); test `tests/test_distributed_checkpoint.py`, `tests/test_checkpoint_retention.py`.

**Interfaces:** Document exact opt-in config for future runs and artifact contents/costs, offline sync caveat, and how to inspect best/last aliases; do not auto-enable policy for an existing run because it changes config fingerprint.

- [ ] Run targeted distributed/retention tests, full `uv run --extra test pytest -q`, compile check, and `git diff --check`.
- [ ] Inspect local step-31/36 pointers and state read-only; do not log or upload them to W&B.
- [ ] Document outcomes and any remaining interrupted/hard-kill limitation.

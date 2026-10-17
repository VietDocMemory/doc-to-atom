# Tuning, Readiness Tests, and Operator Documentation Implementation Plan
> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide a resumable validation-only hyperparameter search, enforce readiness through targeted tests, and publish complete audit/run guides with exact commands for Molab.

**Architecture:** Optuna launches isolated train/evaluate subprocess trials using the stable CLIs and SQLite storage. A final audit generator inventories static findings and runtime artifacts. Documentation distinguishes facts, reconstruction assumptions, and experiments, and links every readiness claim to a command or artifact.

**Tech Stack:** Optuna, subprocess/JSON, SQLite, pytest markers, Markdown, W&B offline/online sync.

**Spec:** [Full Pipeline Readiness Design](../specs/2026-10-17-full-pipeline-readiness-design.md)

## Global Constraints

- Tune on validation only; test metrics never enter objectives or pruning.
- Every trial has an isolated output directory, resolved config, manifest, checkpoint, predictions, metrics, and W&B run.
- Search-space provenance and failed/pruned trial reasons are durable.
- Documentation must label `verified`, `assumption`, `deviation`, or `not implemented` without metric-driven architectural edits.

## Review Focus

- Trial leakage/collisions, subprocess failure propagation, resume behavior, objective direction, and commands that work from a clean clone/Molab shell.

---

## Task 1: Add an Optuna subprocess tuning framework

**Files:**
- Create: `src/doc_to_atom/tuning/__init__.py`
- Create: `src/doc_to_atom/tuning/cli.py`
- Create: `src/doc_to_atom/tuning/runner.py`
- Create: `configs/tuning/default.yaml`
- Modify: `pyproject.toml`
- Test: `tests/test_tuning.py`

- [ ] Write fake-subprocess tests for sampling loss weights, LR, rank/alpha, memory layers, K/temperature, sparse density, and refusal calibration.
- [ ] Implement SQLite-backed study create/resume, deterministic seeds, per-trial overlay configs, timeout/failure handling, pruning, and validation metric extraction.
- [ ] Log study/trial config and artifacts to W&B; support fully offline runs and later `wandb sync`.
- [ ] Add `doc2atom-tune` entry point.
- [ ] Run `py -m uv run pytest tests/test_tuning.py -q`.
- [ ] Commit: `feat(tuning): add resumable validation-only Optuna search`.

## Task 2: Standardize the readiness test matrix

**Files:**
- Modify: `pyproject.toml`
- Modify: `tests/conftest.py`
- Modify: `.github/workflows/ci.yml` (create if absent)
- Test: `tests/test_injection_system.py`
- Test: `tests/test_gradient_flow.py`
- Test: `tests/test_resume_equivalence.py`
- Test: `tests/test_router_composer.py`
- Test: `tests/test_forced_gold.py`
- Test: `tests/test_one_example_overfit.py`
- Test: `tests/test_small_heldout_evaluation.py`

- [ ] Register `unit`, `integration`, `gpu`, `network`, and `molab` markers and make unregistered markers errors.
- [ ] Ensure requested coverage exists for exact LoRA injection, trainable/frozen gradients, resume equivalence, routing/null policy, forced-gold, overfit, and held-out evaluation.
- [ ] Add Windows CPU CI for unit tests and Linux smoke CI for packaging/CLI help; keep GPU/Molab tests opt-in.
- [ ] Run `py -m uv run pytest -q -m "not gpu and not network and not molab"`.
- [ ] Commit: `test: enforce full pipeline readiness matrix`.

## Task 3: Generate and finalize the implementation audit

**Files:**
- Create: `scripts/audit_implementation.py`
- Create: `IMPLEMENTATION_AUDIT.md`
- Test: `tests/test_audit_script.py`

- [ ] Write a fixture test for detection of TODO/NotImplemented, missing referenced artifacts, config conflicts, entry-point import failures, manifest gaps, and stale audit claims.
- [ ] Generate machine-readable `implementation_audit.json` and render Markdown with severity, evidence path, disposition, and verification command.
- [ ] Manually annotate paper-like facts versus reconstruction assumptions and deliberate non-implementations such as Micro-KV.
- [ ] Run `py -m uv run python scripts/audit_implementation.py --check` and its tests.
- [ ] Commit: `docs: add reproducible implementation audit`.

## Task 4: Publish operator guides and exact commands

**Files:**
- Create: `PRE_FULL_RUN_CHECKLIST.md`
- Create: `FULL_TRAINING_GUIDE.md`
- Create: `TUNING_GUIDE.md`
- Modify: `README.md`

- [ ] Document clean setup, local offline smoke, atomization, train, resume, validation, locked test evaluation, tuning, W&B sync, checkpoint recovery, and Molab Accelerate/DeepSpeed commands.
- [ ] Include storage/VRAM estimates as explicitly measured-or-assumed fields, not promises.
- [ ] Add a go/no-go checklist tied to manifests, hashes, cached atomization coverage, resume drill, overfit, held-out metrics, disk capacity, credentials, and W&B mode.
- [ ] Add architecture/deviation tables for gradient checkpointing, routing suppression, Micro-KV, config values, and missing paper artifacts.
- [ ] Verify every documented command with `--help` or a tiny/offline execution; do not run full training.
- [ ] Commit: `docs: publish full training evaluation and tuning guides`.

## Task 5: Final verification and release candidate snapshot

**Files:**
- Modify only files required by verification failures.

- [ ] Run formatting/lint/type checks configured by the repository.
- [ ] Run all non-GPU/non-network tests from a clean process.
- [ ] Run tiny fresh-train, interrupted resume, validation evaluation, score-only evaluation, and one dry-run tuning trial offline.
- [ ] Compare generated audit/checklist claims with artifacts and update any mismatch explicitly.
- [ ] Capture `git status`, resolved dependency versions, test counts, commands, and remaining known limitations in the final report.
- [ ] Commit: `chore: finalize pre-full-run release candidate`.

## Acceptance Commands

```powershell
py -m uv run doc2atom-tune --study doc2atom-smoke --storage sqlite:///outputs/tuning/doc2atom-smoke.db --config configs/tuning/default.yaml --n-trials 1 --dry-run --wandb-mode offline
py -m uv run python scripts/audit_implementation.py --check
py -m uv run pytest -q -m "not gpu and not network and not molab"
```


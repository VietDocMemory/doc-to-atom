# End-to-End Evaluation Implementation Plan
> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the score-only CLI with a resumable checkpoint-to-predictions evaluator covering memory construction, routing, dynamic LoRA composition, generation, retrieval/answer/refusal metrics, and operational statistics.

**Architecture:** Build an `Evaluator` orchestration layer from existing `build_memory`, `DocToAtomSystem`, router, composer, and injector primitives. Keep scoring usable as a separate subcommand. Emit per-example trace JSONL first, then aggregate deterministic metrics and a human-readable report.

**Tech Stack:** PyTorch, Transformers generation, existing Doc-to-Atom modules, rouge-score, JSONL, W&B through the run tracker.

**Spec:** [Full Pipeline Readiness Design](../specs/2026-10-17-full-pipeline-readiness-design.md)

## Global Constraints

- Paper-like configs retain top-K softmax. Null/abstain routing is an opt-in reconstruction deviation.
- Evaluate validation during development; reserve test labels for a final locked run.
- Warmup requests are excluded from latency summaries and timing boundaries synchronize CUDA.
- Predictions are append-only and resumable by stable example ID.

## Review Focus

- Gold-answer normalization, multi-reference scoring, leakage between validation/test, forced-gold isolation, generation determinism, and trace completeness.

---

## Task 1: Make metrics dataset-aware and multi-reference safe

**Files:**
- Modify: `src/doc_to_atom/evaluation/metrics.py`
- Create: `src/doc_to_atom/evaluation/aggregation.py`
- Test: `tests/test_metrics.py`

- [ ] Write failing tests for SQuAD normalization, DROP numeric alternatives, multi-reference max F1/ROUGE-L, explicit refusal labels, Recall@K, MRR, empty gold sets, and macro/per-dataset aggregation.
- [ ] Implement pure metric functions and confidence intervals/counts without importing model code.
- [ ] Separate refusal classification metrics from answer token overlap.
- [ ] Run `py -m uv run pytest tests/test_metrics.py -q`.
- [ ] Commit: `feat(eval): add robust answer retrieval and refusal metrics`.

## Task 2: Add memory provenance and deterministic load/build

**Files:**
- Modify: `src/doc_to_atom/inference/build_memory.py`
- Modify: `src/doc_to_atom/compiler/memory.py`
- Create: `src/doc_to_atom/inference/memory_store.py`
- Test: `tests/test_memory_store.py`

- [ ] Write failing tests for cache reuse, invalidation on checkpoint/record/config hash, device remapping, corrupt cache rejection, and deterministic atom ordering.
- [ ] Store schema version, record hash, compiler/checkpoint hash, dtype, target shapes, atom IDs, prompt/dataset manifest hashes, and tensor checksums.
- [ ] Implement atomic per-record memory cache writes and an index manifest.
- [ ] Run `py -m uv run pytest tests/test_memory_store.py -q`.
- [ ] Commit: `feat(inference): make atom memory cached and provenance checked`.

## Task 3: Build the evaluator orchestration path

**Files:**
- Create: `src/doc_to_atom/evaluation/evaluator.py`
- Create: `src/doc_to_atom/evaluation/generation.py`
- Modify: `src/doc_to_atom/evaluation/logging.py`
- Modify: `src/doc_to_atom/training/system.py`
- Test: `tests/test_evaluator.py`
- Test: `tests/test_forced_gold.py`

- [ ] Write a tiny fake-model test for checkpoint load -> memory -> route -> compose -> activate LoRA -> generate -> trace.
- [ ] Add normal-routing and forced-gold modes with separate names and outputs; forced-gold must bypass retrieval but use identical composition/generation.
- [ ] Record selected IDs/scores/weights, refusal score/decision, gate values, adapter norms, composed ΔW norms/sparsity, token counts, latency breakdown, and peak VRAM.
- [ ] Ensure dynamic adapter activation is exception-safe and cannot leak across examples.
- [ ] Add deterministic sharding and skip-completed resume behavior.
- [ ] Run `py -m uv run pytest tests/test_evaluator.py tests/test_forced_gold.py -q`.
- [ ] Commit: `feat(eval): add checkpoint-to-generation evaluator`.

## Task 4: Expose evaluate, score, validate, and test-lock CLI modes

**Files:**
- Modify: `src/doc_to_atom/evaluation/cli.py`
- Modify: `pyproject.toml`
- Create: `src/doc_to_atom/evaluation/report.py`
- Test: `tests/test_evaluation_cli.py`
- Test: `tests/test_small_heldout_evaluation.py`

- [ ] Write CLI tests for `run`, `score`, cache selection, prediction resume, validation split, and guarded test split.
- [ ] Require `--unlock-test` plus a clean validation report hash for test evaluation; record this decision in the manifest.
- [ ] Produce `predictions.jsonl`, `metrics.json`, `report.md`, and W&B summary/artifacts.
- [ ] Add bounded held-out integration fixture covering relevant and irrelevant questions.
- [ ] Run `py -m uv run pytest tests/test_evaluation_cli.py tests/test_small_heldout_evaluation.py -q` and the full non-GPU suite.
- [ ] Commit: `feat(eval): expose resumable validation and locked test CLI`.

## Task 5: Add optional irrelevant-query null routing as a declared deviation

**Files:**
- Modify: `src/doc_to_atom/router/model.py`
- Modify: `src/doc_to_atom/composer/model.py`
- Modify: `configs/reconstruction/molab_zero2.yaml`
- Create: `configs/experiments/null_router.yaml`
- Test: `tests/test_router_composer.py`

- [ ] Write tests proving legacy paper-like softmax remains bitwise unchanged and null routing can assign zero effective atom mass.
- [ ] Add an opt-in null logit/threshold policy and log null mass/calibration curves; do not enable it in paper-like configs.
- [ ] Calibrate threshold on validation only and serialize calibration provenance.
- [ ] Run `py -m uv run pytest tests/test_router_composer.py tests/test_evaluator.py -q`.
- [ ] Commit: `feat(router): add opt-in null route for irrelevant suppression`.

## Acceptance Commands

```powershell
py -m uv run doc2atom-evaluate run --config configs/experiments/small_overfit.yaml --checkpoint outputs/ci-train/checkpoints/best --data data/squad_small_atoms_openai.jsonl --split validation --output outputs/ci-eval --wandb-mode offline
py -m uv run doc2atom-evaluate score --predictions outputs/ci-eval/predictions.jsonl --output outputs/ci-eval/metrics-rescored.json
py -m uv run pytest tests/test_small_heldout_evaluation.py -q
```


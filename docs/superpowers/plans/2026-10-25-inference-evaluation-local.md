# Local Inference and Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce short, auditable QA answers and comparable SQuAD-style, refusal, routing and provenance reports without changing the model or training objective.

**Architecture:** A shared generation policy formats prompts and stops after one answer line. Evaluation consumes immutable source references, computes metrics from predictions plus explicit routing evidence, and writes versioned run artifacts. A strict comparison path rejects incompatible datasets/protocols.

**Tech Stack:** Python 3.12+, PyTorch/Transformers, `rouge-score`, Pydantic dataset models, pytest.

**Spec:** `docs/superpowers/specs/2026-10-25-inference-evaluation-protocol-design.md`

## Global Constraints

- Only local code/testing; no Molab, GitHub push, cloud upload, paid API, or full-model training.
- Keep source answers exact and checkpoint config-fingerprint validation active.
- No architecture, training loss, optimizer or curriculum change; first-line F1 is diagnostic only.
- Historical step-28 predictions are not a controlled inference comparison against local step-31/36 checkpoint.

## Review Focus

- Leading newline before an answer must not produce an empty prediction; Task 3 tests it.
- A token containing answer, newline and explanation must stop at the answer boundary; Task 3 tests it.
- Prediction rows with duplicate/unknown IDs or changed gold answers must not be silently joined to local atoms; Task 2 tests it.
- K=8 against nine atoms must expose saturation and Recall@1/4, not just a flattering Recall@8; Task 2 tests it.
- Metrics with no answerable or no irrelevant rows must have documented denominator behavior; Task 1 tests it.

---

### Task 1: Answer and refusal metrics

**Files:** Modify `src/doc_to_atom/evaluation/metrics.py`, `src/doc_to_atom/evaluation/cli.py`; test `tests/test_metrics.py`, `tests/test_evaluation_cli.py`.

**Interfaces:** `squad_normalize(text: str) -> str`, `word_f1(prediction: str, gold: str) -> float`, `exact_match(...) -> float`, `first_line(text: str) -> str`, `is_refusal(prediction: str, literal: str = "unanswerable") -> bool`, and `score_rows(rows: list[dict]) -> dict`.

- [ ] Write tests with hand-derived SQuAD article/punctuation/whitespace examples, duplicate and distinct references, full-text versus first-line F1, refusal TP/FP/FN/TN, and zero denominators. Assert `f1` and `raw_f1` use complete prediction, not first-line text.
- [ ] Run `uv run --extra test pytest tests/test_metrics.py tests/test_evaluation_cli.py -q`; confirm new tests fail for missing/wrong behavior.
- [ ] Implement SQuAD F1/EM max-over-gold, separate ROUGE-L max-over-gold, binary refusal precision/recall/F1 and false rates. Preserve existing result keys with the changed refusal semantics explicitly versioned.
- [ ] Re-run targeted tests to green and `git diff --check`.

### Task 2: Routing evidence and legacy score-only join

**Files:** Modify `src/doc_to_atom/evaluation/cli.py`; test `tests/test_evaluation_cli.py`.

**Interfaces:** `routing_scores(ranked_ids: list[str], gold_ids: list[str], candidate_count: int, ks: tuple[int, ...] = (1, 4, 8)) -> dict`; `generate_rows(...)` emits `gold_atom_ids`, `atom_count`, `candidate_count`; score-only CLI accepts `--predictions --data --source` for verified enrichment.

- [ ] Write tests for exact ID/question/answers joins; missing/duplicate/unknown IDs hard fail; three K values/MRR with hand-picked ranks; absent gold IDs yield `null`, not a guessed value; K saturation is reported.
- [ ] Run targeted evaluation tests and observe RED.
- [ ] Implement routing metrics from ranked IDs and gold IDs, legacy enrichment via validated atomized/source JSONL, and per-row counts. Do not reuse historical `recall_at_k` to fabricate other K values.
- [ ] Re-run targeted tests to green.

### Task 3: Shared one-line generation policy

**Files:** Create `src/doc_to_atom/inference/answer_format.py`; modify `src/doc_to_atom/inference/query.py`, `src/doc_to_atom/evaluation/baselines.py`; test `tests/test_query_generation.py`, `tests/test_evaluation_baselines.py`.

**Interfaces:** `build_qa_prompt(question: str, *, context: str | None = None) -> str`, `generate_short_answer(model, tokenizer, prompt: str, *, max_new_tokens: int) -> dict` returning `prediction`, `raw_generation` and stop metadata. Query routing/LoRA remain unchanged. Doc-to-LoRA keeps its baseline-specific prompt but shares stopping/decoding settings.

- [ ] Write tests against deterministic toy tokenizer/model continuations: concise answer, leading newline, newline-plus-explanation in one token, EOS, max-token cap, and refusal literal; assert no gold-aware extraction.
- [ ] Run the new tests and observe RED.
- [ ] Implement greedy generation with shared stop criterion and deterministic answer-boundary handling; wire it into QueryEngine and compatible baselines without touching training prompts.
- [ ] Re-run targeted tests to green.

### Task 4: Evaluation artifacts, provenance and comparison

**Files:** Modify `src/doc_to_atom/evaluation/cli.py`, `notebook.py`; create `src/doc_to_atom/evaluation/compare.py`; update `pyproject.toml`; test `tests/test_evaluation_cli.py`, `tests/test_evaluation_compare.py`, `tests/test_molab_notebook_cells.py`.

**Interfaces:** `doc2atom-evaluate` writes predictions under caller-provided `--output` (not silently moved), metrics under `metrics/`, report under `reports/`; notebook defaults `PRED_EVAL_PATH` to `outputs/molab-single-gpu/predictions/validation.jsonl`. Optional `--wandb-mode disabled|offline|online` defaults to disabled and logs evaluation artifacts only when selected. `doc2atom-compare --input NAME=PATH ... --output PATH` checks common question/gold/dataset/scoring/stop protocol and reports differences; baseline prompt IDs may differ.

- [ ] Write tests for output paths, checkpoint step/config and SHA-256 source+atomized fingerprints, generation settings, non-overwrite, opt-in offline evaluation logging, incompatible comparison rejection, compatible comparison and legacy-row labeling.
- [ ] Run targeted tests and observe RED.
- [ ] Implement versioned row/metrics/report metadata, opt-in evaluation artifact logging and strict comparison; keep config fingerprint verification at checkpoint load. Update notebook destination only; do not move/delete existing root predictions.
- [ ] Re-run targeted tests to green.

### Task 5: Local end-to-end smoke and before/after report

**Files:** Create `tests/test_evaluation_smoke.py`; modify `tests/test_evaluation_cli.py` only if test seams require it; add `docs/evaluation-local-results.md` with evidence and limitations.

**Interfaces:** A CPU toy fixture trains one bounded step with existing `run_training`, reads a committed checkpoint through the real loader with expected fingerprint, generates at least one answerable and one irrelevant prediction, and scores/writes JSONL plus metrics/report. No external model/API download.

- [ ] Write integration test and run it RED before production changes needed by it.
- [ ] Implement only the minimal seams needed for the real train→checkpoint→evaluate path; verify changed fingerprint rejects load.
- [ ] Run smoke, full `uv run --extra test pytest -q`, syntax checks and `git diff --check`.
- [ ] Re-score the immutable supplied predictions locally; report old and new scorer results separately from generation, the 9-atom saturation, unmatched step 28/31 caveat and absence of a real after-generation benchmark.

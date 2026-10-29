# Atomization and Full-Dataset Preparation Implementation Plan
> Historical, superseded proposal: multi-provider/OpenAI commands below are not
> supported by the current MiniMax-only CLI. Do not execute as a current runbook;
> see `../../../README.md` and `../../MINIMAX_MIGRATION_PAPER_AUDIT.md`.
> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn atomization into a provider-neutral, cached, resumable and validated pipeline, then expose reproducible preparation for SQuAD, DROP, ROPES, 2WikiMultiHopQA, QASPER, and LongBench.

**Architecture:** Split transport from the three semantic stages, cache raw provider responses independently, parse without mutating the raw artifact, and canonicalize only after structural/source-span/semantic validation. Dataset adapters map upstream rows into a single raw record schema and write a pinned manifest.

**Tech Stack:** Pydantic, defusedxml/ElementTree-compatible parsing, OpenAI-compatible providers, Hugging Face datasets/Hub, JSONL, SHA-256, pytest.

**Spec:** [Full Pipeline Readiness Design](../specs/2026-10-17-full-pipeline-readiness-design.md)

## Global Constraints

- Cache keys include provider, resolved model, sampling parameters, stage, prompt version/hash, and normalized input hash.
- Never interpolate unescaped source text into XML.
- Preserve raw response bytes and validation diagnostics; canonical output is a separate artifact.
- Offline resume must make progress from cache without provider credentials.
- Dataset revisions and split selections must be recorded, not inferred later.

## Review Focus

- XML entity safety, source-span fidelity, retry idempotency, duplicate IDs, dataset answer semantics, and no hidden canonical repairs.

---

## Task 1: Add a provider protocol and retry policy

**Files:**
- Create: `src/doc_to_atom/atomization/providers.py`
- Create: `src/doc_to_atom/atomization/retry.py`
- Modify: `src/doc_to_atom/atomization/client.py`
- Test: `tests/test_atomization_provider.py`

- [ ] Write fake-provider tests for request/response metadata, transient retry with bounded exponential backoff/jitter, permanent failure, rate limiting, and credential-free cache replay.
- [ ] Define `AtomizationProvider.complete(request) -> ProviderResponse`; implement OpenAI-compatible provider adapters without embedding semantic stages.
- [ ] Move retry classification and budgets into config; persist attempt diagnostics.
- [ ] Run `py -m uv run pytest tests/test_atomization_provider.py -q`.
- [ ] Commit: `refactor(atomize): separate provider transport and retry policy`.

## Task 2: Version prompts and add stage-level cache/resume

**Files:**
- Modify: `src/doc_to_atom/atomization/prompts.py`
- Create: `src/doc_to_atom/atomization/cache.py`
- Modify: `src/doc_to_atom/atomization/client.py`
- Modify: `src/doc_to_atom/atomization/cli.py`
- Test: `tests/test_atomization_cache.py`

- [ ] Write failing tests for cache-key stability, invalidation, interrupted three-stage resume, `--retry-failed`, `--force-stage`, and raw-response preservation.
- [ ] Introduce immutable prompt IDs/versions and prompt SHA-256 in every stage record.
- [ ] Implement atomic cache entries for decomposition, irrelevant probes, and question annotation.
- [ ] Add CLI progress/summary and deterministic record ordering.
- [ ] Run `py -m uv run pytest tests/test_atomization_cache.py -q`.
- [ ] Commit: `feat(atomize): add versioned stage cache and resume`.

## Task 3: Harden XML parsing and validation boundaries

**Files:**
- Modify: `src/doc_to_atom/atomization/parser.py`
- Modify: `src/doc_to_atom/atomization/validation.py`
- Modify: `src/doc_to_atom/data/semantic_validation.py`
- Modify: `src/doc_to_atom/atomization/models.py`
- Test: `tests/test_atomization.py`
- Test: `tests/test_semantic_validation.py`

- [ ] Add hostile/malformed XML tests, escaped-document tests, duplicate ID tests, source offset tests, partial recovery tests, and mutation-separation tests.
- [ ] Parse with entity expansion disabled and resource limits; wrap source payloads safely.
- [ ] Represent validation findings as structured severity/code/location records.
- [ ] Split `raw parsed`, `validated`, and `canonical` models/serialization; make any repair explicit in provenance.
- [ ] Add exact source offsets where possible and normalized-whitespace evidence otherwise.
- [ ] Run `py -m uv run pytest tests/test_atomization.py tests/test_semantic_validation.py -q`.
- [ ] Commit: `fix(atomize): harden XML and make validation non-destructive`.

## Task 4: Build the dataset registry and six adapters

**Files:**
- Create: `src/doc_to_atom/data/registry.py`
- Create: `src/doc_to_atom/data/adapters/__init__.py`
- Create: `src/doc_to_atom/data/adapters/squad.py`
- Create: `src/doc_to_atom/data/adapters/drop.py`
- Create: `src/doc_to_atom/data/adapters/ropes.py`
- Create: `src/doc_to_atom/data/adapters/wiki2.py`
- Create: `src/doc_to_atom/data/adapters/qasper.py`
- Create: `src/doc_to_atom/data/adapters/longbench.py`
- Modify: `src/doc_to_atom/data/prepare.py`
- Test: `tests/test_dataset_adapters.py`

- [ ] Create local row fixtures and failing contract tests for stable IDs, split mapping, context, multi-answer representation, question type, supporting evidence, and metadata.
- [ ] Implement a streaming-capable adapter protocol and dataset registry using the pinned dataset IDs/revisions from the design spec.
- [ ] Handle QASPER unanswerable/multiple annotations, DROP numeric/spans, 2Wiki supporting facts, and LongBench task subdatasets explicitly.
- [ ] Add bounded `--max-records`, deterministic sharding, and offline `--input-cache` support.
- [ ] Run `py -m uv run pytest tests/test_dataset_adapters.py -q`.
- [ ] Commit: `feat(data): add reproducible adapters for six benchmark families`.

## Task 5: Emit and validate dataset manifests

**Files:**
- Create: `src/doc_to_atom/data/manifest.py`
- Modify: `src/doc_to_atom/data/prepare.py`
- Modify: `src/doc_to_atom/data/validation_cli.py`
- Test: `tests/test_dataset_manifest.py`

- [ ] Write tests for source revision, builder/config, split/counts, schema/prompt versions, file hashes, validation summaries, shard inventory, and manifest mismatch detection.
- [ ] Emit `dataset_manifest.json` beside every raw and atomized dataset.
- [ ] Make train/eval reject mismatched or absent manifests by default, with a recorded development-only override.
- [ ] Run `py -m uv run pytest tests/test_dataset_manifest.py -q` and full non-network suite.
- [ ] Commit: `feat(data): add pinned manifests and validation gate`.

## Acceptance Commands

```powershell
py -m uv run python -m doc_to_atom.data.prepare --dataset squad --split train --revision 7b6d24c --max-records 16 --output data/prepared/squad/train.jsonl
py -m uv run doc2atom-atomize --input data/prepared/squad/train.jsonl --output data/atomized/squad/train.jsonl --cache-dir outputs/atom-cache --provider openai-compatible --resume
py -m uv run doc2atom-validate data/atomized/squad/train.jsonl --manifest data/atomized/squad/dataset_manifest.json --semantic
py -m uv run pytest tests/test_dataset_adapters.py tests/test_atomization_cache.py -q
```


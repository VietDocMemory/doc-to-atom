# Atomization Data Contract v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make atomization preserve exact source answers, join provider role annotations to an authoritative question set by ID, repair role overlap transparently, and certify the resulting v2 data before a larger canary.

**Architecture:** Source questions remain the sole owner of `question_id`, `question`, and `answers`; a restricted provider DTO supplies only classification and atom roles. After generated irrelevant probes are added, that combined immutable collection becomes the authoritative intermediate question set and the provider response must form an exact one-to-one ID mapping to it. Canonicalization records every deterministic repair, while structural validation separately reports defects seen in provider output and defects remaining in canonical output.

**Tech Stack:** Python 3.12, Pydantic v2, pytest, XML ElementTree, JSONL, existing OpenAI-compatible provider client.

**Spec:** `docs/superpowers/specs/2026-10-19-atomization-data-contract-v2-design.md`

## Global Constraints

- Do not change model architecture, loss architecture, routing architecture, or compiler behavior.
- Do not run full atomization, training, evaluation, or tuning.
- Write every regression test first and observe the expected failure before changing production code.
- Preserve exact source answer list contents, ordering, duplicates, capitalization, punctuation, and whitespace.
- Treat source JSONL as authoritative during migration; never promote a legacy atomized answer to verified source gold.
- Make missing, duplicate, and unknown provider `question_id` values hard record errors.
- Keep raw provider responses byte-for-byte equivalent as decoded response strings and separate from canonical data.
- Every deterministic repair must emit a stable structured transformation code.
- The default refusal literal remains a training/evaluation policy value, never a normal gold answer.
- Grounding behavior is selected through the dataset adapter; no validator-level global dataset allowlist is permitted.
- Preserve unrelated working-tree changes, including current `.gitignore` and data-file changes.

## Review Focus

- A provider response that repeats one authoritative ID while omitting another must fail rather than satisfy a set-only check; Task 3 tests multiplicity and exact bijection.
- Provider unknown atom references must remain visible after canonical repair makes canonical unknown references zero; Tasks 4 and 6 test both counters independently.
- Re-running canonical validation must not append duplicate transformations or alter source-owned values; Task 4 tests idempotence.
- Legacy records loaded without source JSONL must remain explicitly uncertified and fail the preservation gate; Task 5 tests provenance state.
- Dataset adapters with non-span grounding policy must not receive accidental SQuAD-style verbatim enforcement; Task 6 tests adapter-owned policy dispatch.

---

### Task 1: Introduce schema-v2 source-owned question models

**Files:**
- Modify: `src/doc_to_atom/atomization/models.py`
- Modify: `tests/test_atomization.py`
- Create: `tests/test_atomization_schema_v2.py`

**Interfaces:**
- Produces: `ATOMIZATION_SCHEMA_VERSION = 2`.
- Produces: `QuestionAnnotation.answers: tuple[str, ...]`, serialized as a JSON array.
- Produces: frozen `question_id`, `question`, and `answers` fields.
- Produces: `answer_provenance: Literal["source", "generated_irrelevant", "legacy_unverified"]`.
- Produces: optional compatibility fields `refusal_target` and `refusal_target_derived`, with canonical new records defaulting to null/false.
- Produces: `TransformationEvent` and `AnnotationMeta.transformations` for structured repairs.
- Produces: `StructuredRecord.schema_version == 2` and legacy-v1 input normalization.

- [ ] **Step 1: Write failing schema tests**

Add tests proving JSON with `answers: ["Gold", "Gold", " gold "]` round-trips exactly, assignments to all three source-owned fields fail, irrelevant canonical questions serialize `answers: []`, and a legacy singular `answer` loads only with `answer_provenance == "legacy_unverified"`. Assert serialized v2 output contains `answers` and does not contain `answer`.

- [ ] **Step 2: Run the schema tests and verify RED**

Run: `uv run pytest tests/test_atomization_schema_v2.py -q`

Expected: FAIL because schema v2 fields, immutability, provenance, and transformations do not exist.

- [ ] **Step 3: Implement the minimal schema-v2 models and legacy pre-validator**

Use Pydantic field freezing plus an immutable tuple for answers. Legacy normalization may read singular `answer`, but must mark it unverified; it must never mark legacy answers as source-owned. Keep compatibility loading narrow and remove singular `answer` from canonical serialization.

- [ ] **Step 4: Update existing model fixtures and verify GREEN**

Run: `uv run pytest tests/test_atomization_schema_v2.py tests/test_atomization.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/doc_to_atom/atomization/models.py tests/test_atomization.py tests/test_atomization_schema_v2.py
git commit -m "feat(atomization): add source-owned schema v2 questions"
```

### Task 2: Restrict provider question annotation to a role-only DTO

**Files:**
- Modify: `src/doc_to_atom/atomization/models.py`
- Modify: `src/doc_to_atom/atomization/parser.py`
- Modify: `src/doc_to_atom/atomization/prompts.py`
- Modify: `tests/test_atomization.py`
- Create: `tests/test_question_role_annotation.py`

**Interfaces:**
- Produces: `QuestionRoleAnnotation` with `extra="forbid"` and only `question_id`, optional `question_type`, and three role-ID tuples.
- Produces: `parse_question_roles_xml(xml: str) -> tuple[list[QuestionRoleAnnotation], list[ParseNotice]]`.
- Produces: question-annotation prompt/output schema that contains no `answer`, `answers`, source question text, or refusal target fields.

- [ ] **Step 1: Write failing DTO/parser tests**

Test that constructing the DTO with `answer` or `answers` raises validation errors, XML containing answer-bearing output is rejected with a contract parse notice, and valid role-only XML preserves role order and duplicate IDs for later canonical repair.

- [ ] **Step 2: Run the DTO tests and verify RED**

Run: `uv run pytest tests/test_question_role_annotation.py -q`

Expected: FAIL because the current parser constructs full `QuestionAnnotation` objects and accepts provider answers.

- [ ] **Step 3: Implement the role-only DTO, parser, and prompt contract**

Rename the parser entry point and keep a temporary compatibility alias only if an existing internal caller still requires it. Provider XML must be rejected when forbidden source-owned fields are present; do not silently ignore them.

- [ ] **Step 4: Verify focused parser tests GREEN**

Run: `uv run pytest tests/test_question_role_annotation.py tests/test_atomization.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/doc_to_atom/atomization/models.py src/doc_to_atom/atomization/parser.py src/doc_to_atom/atomization/prompts.py tests/test_atomization.py tests/test_question_role_annotation.py
git commit -m "feat(atomization): restrict provider output to atom roles"
```

### Task 3: Build the authoritative intermediate question set and exact join-by-ID

**Files:**
- Create: `src/doc_to_atom/atomization/question_join.py`
- Modify: `src/doc_to_atom/atomization/client.py`
- Modify: `src/doc_to_atom/atomization/cli.py`
- Create: `tests/test_atomization_join.py`

**Interfaces:**
- Produces: `build_authoritative_questions(source_questions, generated_probes) -> tuple[QuestionAnnotation, ...]`.
- Produces: `join_question_roles(authoritative_questions, role_annotations) -> list[QuestionAnnotation]`.
- Produces: `QuestionJoinError` for duplicate authoritative IDs and missing, duplicate, or unknown provider IDs.
- Consumes: role-only DTOs from Task 2 and source-owned question objects from Task 1.

The authoritative intermediate set is created exactly once after irrelevant-probe generation succeeds or is skipped. It contains all source questions followed by all successfully generated irrelevant probes. From that point through canonical output, this tuple owns IDs, question text, exact answers, irrelevance flags, and generated-probe flags. The role provider must return one and only one annotation for every member; neither source questions nor probes may be added, removed, or replaced during the join.

- [ ] **Step 1: Write failing exact-join tests**

Use hand-built source questions and generated probes. Prove exact source answers survive provider role assignment, generated probes enter the authoritative tuple with empty answers, and missing/duplicate/unknown provider IDs each raise `QuestionJoinError`. Include the repeated-one-ID/omitted-other-ID case.

- [ ] **Step 2: Run join tests and verify RED**

Run: `uv run pytest tests/test_atomization_join.py -q`

Expected: FAIL because the client currently replaces source questions with provider-created objects.

- [ ] **Step 3: Implement authoritative-set construction and join**

Build question/answer input XML with ElementTree escaping. Source answers may be shown to the provider as read-only evidence, but no returned field may write them. Make question-stage contract errors set parse status/error metadata and reject the record instead of emitting atom-only QA output.

- [ ] **Step 4: Verify join and atomizer tests GREEN**

Run: `uv run pytest tests/test_atomization_join.py tests/test_atomization.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/doc_to_atom/atomization/question_join.py src/doc_to_atom/atomization/client.py src/doc_to_atom/atomization/cli.py tests/test_atomization_join.py
git commit -m "fix(atomization): join provider roles to authoritative questions"
```

### Task 4: Enforce disjoint roles with observable deterministic repairs

**Files:**
- Modify: `src/doc_to_atom/atomization/validation.py`
- Modify: `src/doc_to_atom/atomization/models.py`
- Modify: `tests/test_atomization.py`
- Create: `tests/test_role_repair.py`

**Interfaces:**
- Produces: `canonicalize_record(record: StructuredRecord) -> StructuredRecord` with stable transformation events.
- Produces: `provider_unknown_atom_reference_count(record) -> int`, derived from `DROP_UNKNOWN_ATOM_REFERENCE` events.
- Produces: canonical roles satisfying pairwise disjointness and relevant union invariants.

- [ ] **Step 1: Write failing role-repair tests**

Use literal overlapping inputs to verify priority `gold > supporting > distractor`, first-occurrence deduplication, irrelevant-role clearing, relevant-union recomputation, exact transformation codes and before/after details, and idempotence on a second canonicalization pass.

Add a case with two provider unknown IDs and assert provider unknown count remains `2` after canonical lists contain zero unknown IDs.

- [ ] **Step 2: Run role tests and verify RED**

Run: `uv run pytest tests/test_role_repair.py -q`

Expected: FAIL on overlap, missing structured events, and conflated unknown-reference counts.

- [ ] **Step 3: Implement minimal deterministic canonicalization**

Keep `validate_record` as a compatibility wrapper if needed. Each mutation appends exactly one stable event per affected role/ID group and a compatible human-readable warning. Never mutate source-owned fields during role repair.

- [ ] **Step 4: Verify role and existing atomization tests GREEN**

Run: `uv run pytest tests/test_role_repair.py tests/test_atomization.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/doc_to_atom/atomization/models.py src/doc_to_atom/atomization/validation.py tests/test_atomization.py tests/test_role_repair.py
git commit -m "fix(atomization): enforce disjoint atom roles"
```

### Task 5: Add source-authoritative legacy migration

**Files:**
- Create: `src/doc_to_atom/atomization/migration.py`
- Create: `scripts/migrate_atomization_v2.py`
- Modify: `pyproject.toml`
- Create: `tests/test_atomization_migration.py`

**Interfaces:**
- Produces: `migrate_record(legacy_record, source_record) -> StructuredRecord`.
- Produces: `migrate_jsonl(legacy_path, source_path, output_path) -> dict[str, int | float]`.
- Produces: CLI `doc2atom-migrate-v2 --legacy ... --source ... --output ...`.
- Consumes: canonicalization from Task 4.

- [ ] **Step 1: Write failing migration tests**

Create a legacy fixture whose mutated answer disagrees with a source list containing duplicates and punctuation. Assert migrated output exactly equals the source list, not legacy content. Test hard failures for missing records, duplicate source/legacy question IDs, missing source question IDs, unknown legacy question IDs, and non-list/non-string source answers. Assert compatibility-loaded v1 without a source remains `legacy_unverified`.

- [ ] **Step 2: Run migration tests and verify RED**

Run: `uv run pytest tests/test_atomization_migration.py -q`

Expected: FAIL because no authoritative migration exists.

- [ ] **Step 3: Implement migration and CLI**

Accept both legacy source key `answer` and canonical source key `answers` only at the source JSONL boundary, normalize them to an ordered tuple, and never consult the legacy atomized answer for gold content. Carry roles forward through Task 4 canonicalization.

- [ ] **Step 4: Verify migration tests GREEN**

Run: `uv run pytest tests/test_atomization_migration.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/doc_to_atom/atomization/migration.py scripts/migrate_atomization_v2.py pyproject.toml tests/test_atomization_migration.py
git commit -m "feat(atomization): add source-authoritative v2 migration"
```

### Task 6: Extend validator and semantic audit with adapter-owned grounding policy

**Files:**
- Modify: `src/doc_to_atom/data/preprocessors.py`
- Modify: `src/doc_to_atom/data/io.py`
- Modify: `src/doc_to_atom/data/semantic_validation.py`
- Modify: `src/doc_to_atom/data/validation_cli.py`
- Modify: `scripts/audit_semantics.py`
- Modify: `tests/test_semantic_validation.py`
- Create: `tests/test_atomization_acceptance.py`

**Interfaces:**
- Produces: `DatasetAdapter(preprocess, grounding_policy)` and `get_dataset_adapter(name)`.
- Produces: adapter grounding policies `all_answers_verbatim`, `any_answer_verbatim`, and `not_required`.
- Produces: `validate_jsonl(path, source_path=None, reject_span_mismatch=False) -> dict` with distinct `provider_unknown_atom_references` and `canonical_unknown_atom_references`.
- Produces: `audit_jsonl(path, source_path=None) -> dict` without mutation.

Grounding policy is read only from the selected dataset adapter. SQuAD declares `all_answers_verbatim`. Datasets whose extractive semantics have not yet been documented declare `not_required` with an explicit reconstruction note; validator code contains no dataset-name condition.

- [ ] **Step 1: Write failing validator/audit tests**

Test exact source-answer comparison, all three overlap codes, answerable empty gold, irrelevant schema, distinct provider/canonical unknown counts, invalid spans, parse-status failures, and structured transformation counts. Test that a synthetic adapter with `not_required` skips grounding while a synthetic `all_answers_verbatim` adapter emits a hard grounding error for the same record.

- [ ] **Step 2: Run validator tests and verify RED**

Run: `uv run pytest tests/test_semantic_validation.py tests/test_atomization_acceptance.py -q`

Expected: FAIL because current reports omit these invariants and grounding is a global heuristic.

- [ ] **Step 3: Implement adapter policy and validation reports**

The structural validator may canonicalize a copy for reporting but must not hide provider defect counts. The semantic audit must never mutate its input. Add `--source` to validation and semantic-audit CLIs and `--fail-on-error` behavior where absent.

- [ ] **Step 4: Verify validation tests GREEN**

Run: `uv run pytest tests/test_semantic_validation.py tests/test_atomization_acceptance.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/doc_to_atom/data/preprocessors.py src/doc_to_atom/data/io.py src/doc_to_atom/data/semantic_validation.py src/doc_to_atom/data/validation_cli.py scripts/audit_semantics.py tests/test_semantic_validation.py tests/test_atomization_acceptance.py
git commit -m "feat(validation): certify atomization contract v2"
```

### Task 7: Preserve raw provider responses as immutable provenance artifacts

**Files:**
- Create: `src/doc_to_atom/atomization/provenance.py`
- Modify: `src/doc_to_atom/atomization/client.py`
- Modify: `src/doc_to_atom/atomization/cli.py`
- Create: `tests/test_atomization_provenance.py`

**Interfaces:**
- Produces: frozen `RawProviderResponse(record_id, stage, attempt, provider, model, content, sha256)`.
- Produces: `write_raw_response_jsonl(response, path)` using append-only behavior during resume.
- Produces: atomizer `last_raw_responses` preserving every attempt, including failed/retried parsing.
- Produces: CLI option `--raw-responses`, defaulting to `<output>.raw-responses.jsonl`.

- [ ] **Step 1: Write failing provenance tests**

Inject a fake provider transport returning XML with unusual whitespace and entity text. Assert the stored content equals the exact returned string, SHA-256 is computed over its exact UTF-8 bytes, parsing/canonicalization leave it unchanged, retry attempts remain separate, and resume appends without rewriting prior lines.

- [ ] **Step 2: Run provenance tests and verify RED**

Run: `uv run pytest tests/test_atomization_provenance.py -q`

Expected: FAIL because raw responses are currently discarded.

- [ ] **Step 3: Implement response capture and sidecar writing**

Separate the external request method from the capture wrapper so tests replace only network transport. Record content immediately after receipt and before parsing. Flush each sidecar entry so a later record failure does not erase completed provider evidence.

- [ ] **Step 4: Verify provenance tests GREEN**

Run: `uv run pytest tests/test_atomization_provenance.py tests/test_atomization_join.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/doc_to_atom/atomization/provenance.py src/doc_to_atom/atomization/client.py src/doc_to_atom/atomization/cli.py tests/test_atomization_provenance.py
git commit -m "feat(atomization): retain immutable provider responses"
```

### Task 8: Centralize derived training/evaluation target policy

**Files:**
- Create: `src/doc_to_atom/data/targets.py`
- Modify: `src/doc_to_atom/training/loop.py`
- Modify: `scripts/run_heldout_generalization.py`
- Modify: `scripts/run_component_isolation.py`
- Modify: `scripts/run_parametric_transfer_isolation.py`
- Modify: `scripts/fix_known_squad_mapping.py`
- Modify: `tests/test_training_loop.py`
- Create: `tests/test_target_policy.py`

**Interfaces:**
- Produces: `TargetValue(text: str, provenance: Literal["source_answer", "derived_refusal"])`.
- Produces: `resolve_target(qa, refusal_literal="unanswerable") -> TargetValue`.
- Consumes: exact v2 answers and irrelevant flags from Task 1.

- [ ] **Step 1: Write failing target-policy tests**

Assert answerable questions select the first exact source answer under the existing training policy, irrelevant questions map to the configured literal with derived provenance, irrelevant `answers` stay empty, and an answerable question with no answers raises a contract error.

- [ ] **Step 2: Run target tests and verify RED**

Run: `uv run pytest tests/test_target_policy.py -q`

Expected: FAIL because refusal mapping is hard-coded in training/scripts and has no provenance.

- [ ] **Step 3: Implement and adopt the shared target resolver**

Replace direct `qa.answer` and inline `"unanswerable"` selection at training/evaluation-target boundaries. Do not change model execution, losses, metric formulas, or run any training.

- [ ] **Step 4: Verify target and training-loop tests GREEN**

Run: `uv run pytest tests/test_target_policy.py tests/test_training_loop.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/doc_to_atom/data/targets.py src/doc_to_atom/training/loop.py scripts/run_heldout_generalization.py scripts/run_component_isolation.py scripts/run_parametric_transfer_isolation.py scripts/fix_known_squad_mapping.py tests/test_training_loop.py tests/test_target_policy.py
git commit -m "fix(data): derive refusal targets outside source answers"
```

### Task 9: Run the relevant suite and regenerate a bounded canary

**Files:**
- Modify: `src/doc_to_atom/data/prepare.py`
- Create: `scripts/report_atomization_acceptance.py`
- Create: `tests/test_prepare_source_contract.py`
- Create at runtime: `artifacts/atomization_canary_v2/` outputs, ignored unless repository policy explicitly tracks them.

**Interfaces:**
- Produces: prepared source JSONL using `questions[].answers` exactly as received upstream.
- Produces: an acceptance report comparing legacy before metrics and v2 after metrics.

- [ ] **Step 1: Write failing preparation test**

Use a local SQuAD-shaped fixture with duplicate and multi-reference answers. Assert preparation writes exact `answers` lists and no singular `answer` field.

- [ ] **Step 2: Run preparation test and verify RED**

Run: `uv run pytest tests/test_prepare_source_contract.py -q`

Expected: FAIL because preparation currently writes singular `answer`.

- [ ] **Step 3: Implement source preparation and acceptance reporter**

The reporter takes `--before`, `--after`, and `--source`, emits exact answer mismatch/role-overlap counts, both unknown-reference counts, validation/audit summaries, and the acceptance-gate decision.

- [ ] **Step 4: Verify focused and full relevant tests**

Run focused suite:

```bash
uv run pytest tests/test_atomization.py tests/test_atomization_schema_v2.py tests/test_question_role_annotation.py tests/test_atomization_join.py tests/test_role_repair.py tests/test_atomization_migration.py tests/test_semantic_validation.py tests/test_atomization_acceptance.py tests/test_atomization_provenance.py tests/test_target_policy.py tests/test_training_loop.py tests/test_prepare_source_contract.py -q
```

Then run repository suite excluding explicit Molab hardware tests:

```bash
uv run pytest -q -m "not molab"
```

Expected: all tests PASS.

- [ ] **Step 5: Build a one-document canary input without changing tracked/user data**

Create `artifacts/atomization_canary_v2/source.jsonl` from one prepared SQuAD document. Extract the matching legacy atomized record into `artifacts/atomization_canary_v2/legacy-before.jsonl` so before/after metrics cover the exact same source questions. Do not restore, overwrite, or stage the currently deleted `data/squad_small_*` files or untracked `data/squad_molab_*` files.

- [ ] **Step 6: Regenerate only the one-document atomization canary**

Run:

```bash
uv run doc2atom-atomize --provider openai --input artifacts/atomization_canary_v2/source.jsonl --output artifacts/atomization_canary_v2/atoms.jsonl --raw-responses artifacts/atomization_canary_v2/raw-responses.jsonl
```

This is three bounded provider stages for one document, not full atomization. If credentials/network/provider availability block the call, preserve the failure artifact and report it as a canary blocker; do not substitute fabricated production metrics.

- [ ] **Step 7: Validate, audit, and create before/after report**

Run:

```bash
uv run doc2atom-validate --input artifacts/atomization_canary_v2/atoms.jsonl --source artifacts/atomization_canary_v2/source.jsonl --reject-span-mismatch
uv run python scripts/audit_semantics.py --input artifacts/atomization_canary_v2/atoms.jsonl --source artifacts/atomization_canary_v2/source.jsonl --fail-on-error
uv run python scripts/report_atomization_acceptance.py --before artifacts/atomization_canary_v2/legacy-before.jsonl --after artifacts/atomization_canary_v2/atoms.jsonl --source artifacts/atomization_canary_v2/source.jsonl --output artifacts/atomization_canary_v2/acceptance-report.json
```

Expected after metrics: preservation rate `1.0`; answerable empty gold, applicable grounding failures, all overlaps, irrelevant answer/gold/supporting violations, canonical unknown refs, invalid spans, and parse errors all `0`. Provider unknown refs must be reported separately and are a blocker if non-zero even when repaired canonical refs are zero.

- [ ] **Step 8: Commit code and tests only**

```bash
git add src/doc_to_atom/data/prepare.py scripts/report_atomization_acceptance.py tests/test_prepare_source_contract.py
git commit -m "test(atomization): add schema v2 canary acceptance gate"
```

### Task 10: Final verification and acceptance report

**Files:**
- Modify if needed: `docs/IMPLEMENTATION_MAP.md`
- Create: `docs/ATOMIZATION_V2_ACCEPTANCE.md`

**Interfaces:**
- Produces: human-readable report linked to machine-readable canary evidence.

- [ ] **Step 1: Use verification-before-completion and rerun fresh checks**

Run `python -m compileall` on changed source/scripts, the focused suite, the full `not molab` suite, validation, semantic audit, and acceptance reporter. Record commands, exit codes, and exact summaries.

- [ ] **Step 2: Write the acceptance report**

Document files changed, new tests, RED/GREEN evidence, source/legacy/canonical example, before/after metrics, validator/audit results, raw-response path/hash verification, assumptions/deviations, and any blocker before a 50–100 document canary.

- [ ] **Step 3: Check scope and working tree**

Confirm no model architecture/config changes, no training/tuning outputs, no full-data artifacts, and no unrelated user changes were staged. Run `git diff --check`.

- [ ] **Step 4: Commit the report**

```bash
git add docs/IMPLEMENTATION_MAP.md docs/ATOMIZATION_V2_ACCEPTANCE.md
git commit -m "docs: report atomization v2 acceptance"
```


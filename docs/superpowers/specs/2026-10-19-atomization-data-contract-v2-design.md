# Atomization Data Contract v2 Design

## Goal

Prevent semantic corruption during atomization by making source questions and answers authoritative, restricting the LLM annotation stage to atom-role assignment, enforcing disjoint atom roles, and representing irrelevant questions without mixing refusal literals into source gold answers.

This change does not alter model architecture, tune hyperparameters, run full atomization, or run training.

## Current failure evidence

The existing Molab SQuAD sample contains 24 source questions. Exact comparison against the atomized output finds 11 answer-list mismatches, including answer expansion, capitalization changes, loss of duplicate references, and collapse of multi-reference answers. Eight atomized questions assign the same atom to both `gold_atom_ids` and `supporting_atom_ids`. The existing structural validator and semantic audit report zero errors for that sample, so these failures are not currently observable through supported validation commands.

The root cause is the question-annotation boundary: the prompt asks the provider to generate an answer, the parser accepts that answer, and the client replaces source question objects with provider-created question objects. Role post-processing removes unknown IDs but does not make roles disjoint.

## Scope

In scope:

- Version the atomized record contract.
- Preserve source-owned question fields exactly.
- Separate source questions from provider-owned role annotations.
- Add explicit migration behavior for legacy atomized records.
- Enforce and audit role disjointness.
- Normalize irrelevant-question representation.
- Move refusal literal selection into training/evaluation policy.
- Preserve raw provider responses independently from parsed and repaired data.
- Extend structural and semantic validation with machine-readable counts and errors.
- Regenerate only a bounded canary sample and compare before/after metrics.

Out of scope:

- Model architecture changes.
- Hyperparameter tuning.
- Full atomization or full training.
- General refactoring unrelated to the data contract.
- Treating a mutated legacy atomized answer as source truth.

## Contract ownership

`question_id`, `question`, and `answers` are source-owned immutable fields. Their canonical values come from the source JSONL supplied to atomization or migration. Provider output must never replace, normalize, deduplicate, reorder, expand, paraphrase, or canonicalize them.

Answer preservation means exact Python list equality, including string contents, capitalization, punctuation, ordering, and duplicate entries. It is not normalized-text equality and not set equality.

The canonical v2 question shape is:

```json
{
  "question_id": "source-id",
  "question": "source question text",
  "answers": ["verbatim source answer"],
  "question_type": "single_hop",
  "gold_atom_ids": ["atom_1"],
  "supporting_atom_ids": [],
  "distractor_atom_ids": [],
  "relevant_atom_ids": ["atom_1"],
  "is_irrelevant": false,
  "refusal_target": null,
  "refusal_target_derived": false
}
```

The canonical irrelevant shape is:

```json
{
  "question_id": "generated_irrelevant_0",
  "question": "source or generated question text",
  "answers": [],
  "question_type": "irrelevant",
  "gold_atom_ids": [],
  "supporting_atom_ids": [],
  "distractor_atom_ids": [],
  "relevant_atom_ids": [],
  "is_irrelevant": true,
  "refusal_target": null,
  "refusal_target_derived": false
}
```

The literal `unanswerable` is a training/evaluation policy value, not a source gold answer. Canonical v2 data therefore leaves `refusal_target` null by default. If an imported artifact explicitly stores a refusal target for compatibility, the record must mark it as derived with `refusal_target_derived: true`; metrics and answer preservation must still use `answers: []`.

## Schema version and legacy migration

`StructuredRecord` gains an explicit atomization schema version of `2`. The loader remains able to parse schema-v1 records with singular `answer` fields, but this compatibility parser does not certify answer provenance.

Authoritative migration requires both the legacy atomized JSONL and its source JSONL. Migration joins records by `record_id` and questions by `question_id`, then copies `question_id`, `question`, and the exact source answer list into v2 output. Legacy atom roles may be carried forward and repaired, but legacy answers are ignored whenever source data is available.

The migration CLI must fail rather than guess when:

- a legacy record lacks a matching source record;
- a source or legacy record contains duplicate question IDs;
- a legacy question lacks a matching source question;
- a legacy annotation references an unknown source question;
- the source input cannot represent answers as an ordered list of strings.

Direct compatibility loading of a standalone v1 record remains possible for old tests and diagnostic tools, but the resulting record is marked as having unverified answer provenance. It must not be accepted by the source-preservation validation gate used for canary or full-data readiness.

## Role-only provider annotation

The provider-facing annotation DTO contains no `answer`, `answers`, `question`, refusal target, or other source-owned value. It may contain only:

- `question_id`;
- `question_type` when classification is requested;
- `gold_atom_ids`;
- `supporting_atom_ids`;
- `distractor_atom_ids`.

The provider may receive source question text and source answers as read-only evidence for choosing atoms, but the response schema cannot return them. The client joins parsed annotations to source questions by exact `question_id` and constructs canonical question objects from the source objects.

For each atomization record, missing, duplicate, or unknown annotation `question_id` values are hard errors. The record is not emitted as a valid canonical QA record. Provider annotation may not silently drop a question or introduce a new one. Generated irrelevant probes become source-owned for the remainder of that atomization call before role annotation begins.

XML inserted into provider prompts is constructed with an XML library or equivalent escaping rather than raw string interpolation.

## Deterministic role repair

Known atom IDs are deduplicated while preserving first occurrence. Unknown atom references are removed from canonical output and recorded as validation errors/transformations; readiness validation requires the resulting unknown-reference count to be zero.

Role priority is deterministic:

1. Gold keeps all known, first-occurrence IDs.
2. Supporting removes every ID already present in gold.
3. Distractor removes every ID already present in gold or retained supporting.

The postcondition is:

```text
gold ∩ supporting = ∅
gold ∩ distractor = ∅
supporting ∩ distractor = ∅
```

Every mutation emits a structured transformation event with a stable code and before/after details. At minimum the contract defines:

- `DROP_UNKNOWN_ATOM_REFERENCE`
- `DEDUPLICATE_ROLE_ATOM`
- `REMOVE_GOLD_FROM_SUPPORTING`
- `REMOVE_GOLD_FROM_DISTRACTOR`
- `REMOVE_SUPPORTING_FROM_DISTRACTOR`
- `CLEAR_IRRELEVANT_GOLD`
- `CLEAR_IRRELEVANT_SUPPORTING`
- `RECOMPUTE_RELEVANT_UNION`

Human-readable warning strings remain available for backward compatibility, but are derived from or accompanied by structured events. No deterministic repair is silent.

Irrelevant questions always have empty `answers`, `gold_atom_ids`, `supporting_atom_ids`, and `relevant_atom_ids`. The validator applies deterministic repair where safe and records the corresponding transformation. An irrelevant question with source answers is a source-contract error, not an opportunity to reinterpret those answers as refusal text.

## Raw response provenance

Each provider call records the exact returned response string before parsing. Decomposition retries are separate attempts; no retry overwrites the previous response. Raw response artifacts include record ID, stage, attempt number, provider/model identity, exact response content, and SHA-256.

Parsed output, canonical output, and transformations are separate representations. Validation and repair never mutate the retained raw response. A raw-response regression test uses deliberately unusual whitespace and XML text and verifies exact string equality and stable hash after parsing and repair.

For the bounded implementation, the atomization CLI writes a companion raw-response JSONL artifact next to the canonical output. Resume appends new attempts without changing retained prior attempts. Full content-addressed stage caching remains part of the broader pipeline-readiness work and is not introduced here.

## Training and evaluation policy

Training and evaluation resolve a target through a policy function:

```text
if is_irrelevant:
    return configured refusal literal (default: "unanswerable")
else:
    return a source answer selected by the existing multi-reference policy
```

The refusal literal is never inserted into `answers`. Training and evaluation traces identify whether a target is a source answer or a derived refusal-policy target.

This task updates only the target-mapping boundary and its tests. It does not run training or change losses, routing, compiler behavior, or generation architecture.

## Validation and acceptance reporting

Structural validation receives canonical atomized records and, for source-preservation certification, the source JSONL. It reports at least:

- source question count;
- exact answer matches and mismatches;
- answer preservation rate;
- answerable questions with empty gold;
- source answers not grounded in any gold atom for declared span-answer tasks;
- all three role-overlap counts;
- irrelevant questions with answers, gold, or supporting atoms;
- unknown atom references;
- invalid source spans;
- parse errors;
- structured transformation counts by code.

Span grounding is enforced only when the source task or question explicitly declares span-answer semantics. SQuAD canary questions are span-answer questions. For multi-reference questions, every non-empty source answer must appear verbatim in at least one gold atom's `content` or `source_span`; if a dataset later documents alternative references that are not all extractive, its adapter must declare a different grounding policy rather than silently weaken this rule.

The semantic audit is non-mutating. It independently reports role overlap, irrelevant-schema violations, unknown references, empty gold, relevant-union mismatch, exact source-answer mismatch when source records are supplied, and span-grounding failures.

The canary readiness gate requires:

- exact source-answer preservation rate `1.0`;
- zero answerable questions with empty gold;
- zero applicable span-grounding failures;
- zero role overlaps;
- zero irrelevant questions with answers, gold, or supporting atoms;
- zero unknown atom references;
- zero invalid source spans;
- zero parse errors.

Transformation counts may be non-zero because deterministic role repair is allowed, but every count must be attributable to a structured code.

## Test strategy

Regression tests are written and observed failing before production changes. They cover:

- exact preservation of ordered, duplicate, case-sensitive, and punctuated source answer lists;
- provider answer fields being impossible in the annotation DTO;
- source-owned question fields surviving malicious or malformed provider output;
- hard failure for missing, duplicate, and unknown question IDs;
- role-priority repair and each overlap transformation code;
- non-mutating semantic overlap detection;
- irrelevant schema and derived refusal policy mapping;
- v1 compatibility loading and source-authoritative migration;
- rejection of uncertified legacy answers at the preservation gate;
- exact raw-response retention across parsing and repair;
- source-span, atom-reference, parse-status, and SQuAD span-grounding acceptance counts.

After focused tests pass, the full relevant test suite runs. A bounded SQuAD canary is regenerated into a new artifact path, then structural validation and semantic audit run with the source JSONL. No full atomization or training command is executed.

## Before/after report

The final report includes:

- files changed;
- tests added and their red/green evidence;
- a source/legacy/canonical example;
- before/after answer mismatch and role-overlap counts;
- complete validator and semantic-audit summaries;
- raw-response artifact location and integrity result;
- any blocker remaining before a 50–100 document canary.


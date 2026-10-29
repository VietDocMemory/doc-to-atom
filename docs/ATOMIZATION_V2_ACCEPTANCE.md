# Atomization data contract v2 acceptance

> Historical acceptance evidence. Provider/model names and measured observations
> below are preserved, not instructions to use a removed backend. See
> `MINIMAX_MIGRATION_PAPER_AUDIT.md` for the current MiniMax-only implementation.

Date: 2026-10-19
Scope: atomization semantics and data contracts only. No model architecture,
training, tuning, or full-dataset run was changed or executed.

## Result

The one-document SQuAD canary passed every requested contract gate. This is a
bounded integration check, not evidence that a 50--100 document canary or a
full dataset will pass without provider-level failures.

| Gate | Legacy before | Schema-v2 after |
|---|---:|---:|
| Source answer preservation | 1/1 (100%) | 1/1 (100%) |
| Answerable questions without gold atoms | 0 | 0 |
| Applicable grounding failures | 0 | 0 |
| Gold/supporting overlap | 0 | 0 |
| Gold/distractor overlap | 0 | 0 |
| Supporting/distractor overlap | 0 | 0 |
| Irrelevant questions with answers | 0 | 0 |
| Irrelevant questions with gold atoms | 0 | 0 |
| Irrelevant questions with supporting atoms | 0 | 0 |
| Provider unknown atom references | 0 | 0 |
| Canonical unknown atom references | 0 | 0 |
| Invalid source spans | 0 | 0 |
| Parse errors | 0 | 0 |

The legacy row happened to have the same answer text and valid roles, so this
particular before/after sample does not manufacture an improvement. Its answer
is compatibility-loaded as `legacy_unverified`; schema v2 obtains the exact
answer from source JSONL and marks it `source`. The reporter aligns source
questions by `question_id`, excludes unrelated source questions in the same
legacy document, and retains each run's generated irrelevant probes for
irrelevant-schema checks.

## Concrete data flow

For source question `5733be284776f41900661182`:

- Source JSONL owns `question_id`, the exact question text, and
  `answers: ["Saint Bernadette Soubirous"]`.
- Legacy input contains singular `answer: "Saint Bernadette Soubirous"`; this
  value is never promoted to verified source gold by migration.
- The provider question-annotation response contains only the question ID,
  question type, and atom roles.
- Join-by-ID produces the canonical source-owned answer tuple unchanged.
- Canonical output assigns `gold_atom_ids: ["atom_9"]`; atom 9 contains the
  exact source answer. Two generated irrelevant probes have `answers: []`, no
  gold/supporting atoms, and no stored literal refusal answer.
- Canonicalization recorded `RECOMPUTE_RELEVANT_UNION` once. This is an
  observable deterministic transformation, not a silent repair.

## Implemented contract

- `question_id`, `question`, and `answers` are frozen source-owned fields.
- Provider annotation uses an `extra="forbid"` role-only DTO and cannot return
  answers, question text, or refusal targets.
- The authoritative intermediate question tuple is built after generated
  irrelevant probes are added. Provider IDs must form an exact bijection with
  it; missing, duplicate, and unknown IDs are hard errors.
- Source question IDs cannot be empty or synthesized by the CLI. Persisted
  artifacts are audited for missing/duplicate/unknown question IDs and for an
  exact source record set; a missing whole record can no longer inflate the
  answer-preservation rate.
- Role repair uses priority `gold > supporting > distractor`, records stable
  transformation codes, and is idempotent.
- Migration requires authoritative source JSONL for source questions and does
  not trust mutated legacy answers. Explicit legacy generated probes may be
  retained with empty answers.
- Validator output separates provider unknown-reference count (repaired input)
  from canonical unknown-reference count (persisted output).
- Grounding policy belongs to each dataset adapter. SQuAD currently requires
  all answers verbatim; adapters without documented extractive semantics use
  an explicit `not_required` reconstruction policy.
- Refusal literals are derived only at training/evaluation target boundaries.
  They are not normal gold answers.
- Every raw provider response is captured before parsing in an append-only
  JSONL sidecar with a SHA-256 hash over its exact UTF-8 decoded string.

## Tests and TDD evidence

New focused coverage includes schema immutability/exact answer round-trip,
role-only DTO parsing, exact join-by-ID, disjoint role repair/idempotence,
source-authoritative migration, semantic acceptance counters, raw-response
provenance, refusal-target policy, source preparation, and population-aligned
acceptance reporting.

Each production slice was preceded by a failing focused test. The final
acceptance reporter regression specifically failed with `questions == 3`
instead of `2` before its question-ID population filter was implemented.
Final self-review also drove RED/GREEN cases for synthesized/missing IDs,
persisted duplicate/unknown question IDs, acceptance reports that ignored
non-enumerated semantic errors, and missing/duplicate/unknown whole records.

Fresh final verification commands are recorded below; all must exit 0 before
this report is considered current:

```powershell
uv run python -m compileall -q src scripts
uv run pytest tests/test_atomization.py tests/test_atomization_schema_v2.py tests/test_question_role_annotation.py tests/test_atomization_join.py tests/test_role_repair.py tests/test_atomization_migration.py tests/test_semantic_validation.py tests/test_atomization_acceptance.py tests/test_atomization_provenance.py tests/test_target_policy.py tests/test_training_loop.py tests/test_prepare_source_contract.py tests/test_acceptance_report.py -q
uv run pytest -q -m "not molab"
uv run doc2atom-validate --input artifacts/atomization_canary_v2/atoms.jsonl --source artifacts/atomization_canary_v2/source.jsonl --reject-span-mismatch
uv run python scripts/audit_semantics.py --input artifacts/atomization_canary_v2/atoms.jsonl --source artifacts/atomization_canary_v2/source.jsonl --fail-on-error
uv run python scripts/report_atomization_acceptance.py --before D:\Documents\KLTN\doc-to-atom\data\squad_molab_atoms.jsonl --after artifacts/atomization_canary_v2/atoms.jsonl --source artifacts/atomization_canary_v2/source.jsonl --output artifacts/atomization_canary_v2/acceptance-report.json
git diff --check
```

## Canary provenance

The local, intentionally untracked canary directory is
`artifacts/atomization_canary_v2/`. It contains source, canonical atoms,
append-only raw responses, and the machine-readable acceptance report. Three
OpenAI calls used `gpt-4.1-mini-2025-04-14`, one attempt per stage:

| Stage | SHA-256 of exact response content |
|---|---|
| decomposition | `a35b5baaad2df04123664aa8a90cb91385116fec427ad65fe773545d7c728b57` |
| irrelevant probe | `28c7f23a11b6ea60233093cdcd943f9d48e55ff7ccd0c2e036636ad507785fdb` |
| question annotation | `1146a46baf2f0524cd978ec7d93e87faf4483e502ba632020d9e51715cd06258` |

All three stored hashes were recomputed from `content` and matched. Secrets
were loaded process-locally from the original repo `.env`; no secret value was
copied, logged, or committed.

## Assumptions and deviations

- The prompt remains a reconstructed prompt, not a published official prompt.
- This canary used OpenAI instead of the paper-like MiniMax provider; provider
  and pinned model are written into provenance.
- Only SQuAD currently enforces `all_answers_verbatim`. Other dataset adapters
  retain documented `not_required` policies until their grounding semantics
  are specified and tested.
- Existing first-answer training behavior is retained for answerable questions;
  the selected value now has explicit `source_answer` provenance.
- Generated irrelevant probes are authoritative only after probe generation
  for that atomization attempt. Their literal refusal target remains derived.
- Raw provenance preserves the exact decoded provider response string, not the
  lower-level HTTP transport bytes.

## Blockers before a 50--100 document canary

There is no code-contract blocker found by the one-document gate. Operational
readiness still requires valid provider credentials/quota and enough runtime
for three calls per document. Before scaling, run the same validator, semantic
audit, and acceptance reporter; stop on any provider unknown reference, parse
error, span error, grounding failure, role overlap, or source mismatch. A
single passing document is not sufficient to waive those gates.

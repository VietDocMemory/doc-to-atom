# MiniMax annotation pilot: validation repair 01

## Outcome

The combined dataset passes strict source-aware validation. **No training was
started**, no checkpoint metadata was changed, and no GitHub/Molab actions occurred.

- Parent: `outputs/annotation-effort-pilot-20261030/`
- Repair: `outputs/annotation-effort-pilot-20261030-repair-01/`
- Provisional model/depth: `MiniMax-M3.1-Flash`, `reasoning_effort=low`.
- All five original effort trials, prompts, benchmark results and provenance remain
  unchanged. The default change does not rewrite historical settings.
- All four valid training records are byte-identical. Only
  `squad_validation_0` changed.

## Repairs and provenance

1. The probe response began `irrelevant_questions>` and ended with its complete
   closing root. Inserting exactly one leading `<` yields well-formed XML with
   the same two intact questions. No probe-generation retry was needed. The
   original response remains byte-for-byte intact; recovered XML is a separate file.
2. A malformed `</emantic_type>` closing tag in the logo atom was corrected to
   `</semantic_type>`. Three unsupported `participates` edges were explicitly
   **removed, not mapped to another relation**. This recovered the three saved
   atoms `atom_5`, `atom_6`, and `atom_12`; no new atom text was generated.
3. Location atom `atom_8` omitted the source's terminal period. Its source span
   was extended by that one source character, with offsets logged. All original
   source answers, including the long period-terminated variant, are unchanged.
4. Restoring the defeat-event atom exposed a semantic defect the lexical audit
   alone had missed: the old winner label referred to conference-champion facts.
   Exactly one explicit **question_annotation attempt 2** was made for the
   validation record's corrected atom bank and recovered probes. The winner now
   selects `atom_5`; the location selects `atom_8`; both probes are irrelevant.

The retry uses the current role-disjointness prompt, saved in `retry_request.json`,
plus a targeted winner-evidence instruction and the authoritative context/spans.
This is not a second benchmark or a new effort-selection trial. The generation
cap remained 524288 tokens, timeout 1800 seconds, with no automatic retries.

Historical warning strings remain in metadata as history; the successful recovery,
status transition, old/new annotations and canonicalization are recorded in
`transformations.json` and each repaired record's structured transformations.
The original provider response is never relabeled as a repaired response.

## Strict validation

| Combined train + validation | Before | After |
|---|---:|---:|
| Records | 5 | 5 |
| Atoms | 57 | 60 |
| Questions, including probes | 32 | 34 |
| Source questions / answers preserved | 24/24 | 24/24 |
| Errors | 2 | 0 |
| Gold grounding failures | 1 | 0 |
| Parse failures | 1 | 0 |
| Invalid exact source spans | 0 | 0 |
| Gold/supporting/distractor overlaps | 0 | 0 |
| Unknown atom references | 0 | 0 |

The non-mutating strict gate also checks missing/duplicate records and questions,
question text, ordered answer variants, source context/dataset/split equality,
exact source spans, role IDs, relation targets, metadata counts and parser statuses.
No answer normalization or relaxation of the grounding policy was used.

Validation is mechanical consistency and source grounding, not an assertion that
every annotation has been independently human-reviewed. Retained warning strings
are historical diagnostics, not new audit errors.

## Usage and costs

The incremental repair made **1 API call**: 2502 input + 985 completion = **3487
tokens**, including 137 reasoning tokens. Request latency was **8.68 seconds**.
The production response history including that retry records 37705 total tokens.
`usage.json` retains per-effort benchmark usage and production/retry breakdowns.
The selected low pilot responses were reused in production: do not add benchmark
and production totals without deduplicating those shared requests.

Actual billed USD is **unavailable** in the saved API responses; token usage is
not presented as an invoice or an estimated dollar charge.

## Dataset SHA-256 fingerprints

```text
atoms.jsonl
35c182227271519374a0ebefe06d6a0783620ff0118c115975334062b71789f1

raw_source.jsonl
a1f164cb83cbc74cac1eadf74b5566812ab754ecc6dea353f93e40dbf813d556

raw_responses.jsonl
ee4e0a2156d6eb5680653d565113657a8e152346d1f5a2858e801e99cfa1d480
```

`original_fingerprints.json` inventories the untouched parent pilot, including all
five benchmark directories. `fingerprints.json` inventories repair artifacts;
`release.json` records the gate and dataset hashes. The authoritative parent raw
source also retains its original SHA-256 from the parent manifest; the production
source has different line endings but identical parsed source records.

## Artifacts and reproducibility

The repair directory contains combined and split-specific `atoms.jsonl`,
`raw_source.jsonl`, and `raw_responses.jsonl`, before/after audits, the original raw
sidecar copy, the retry-only sidecar, recovered XML, request, transformations,
manifest, usage and fingerprints. Use these repaired files, not the failed parent
production release. Training remains stopped.

```powershell
uv run scripts/validate_data.py `
  --input outputs/annotation-effort-pilot-20261030-repair-01/atoms.jsonl `
  --source outputs/annotation-effort-pilot-20261030-repair-01/raw_source.jsonl `
  --reject-span-mismatch
```

The paid repair runner refuses existing destinations and requires the explicit
`--retry-question-annotation` flag. Do **not** rerun it merely to validate this
already repaired dataset; doing so with a new destination would incur another call.

from __future__ import annotations

import argparse
import json
from pathlib import Path

from doc_to_atom.data.io import read_jsonl
from doc_to_atom.data.semantic_validation import audit_records, load_source_records


REQUIRED_ZERO = (
    "answerable_empty_gold",
    "grounding_failures",
    "gold_supporting_overlap",
    "gold_distractor_overlap",
    "supporting_distractor_overlap",
    "irrelevant_with_answers",
    "irrelevant_with_gold",
    "irrelevant_with_supporting",
    "provider_unknown_atom_references",
    "canonical_unknown_atom_references",
    "parse_errors",
)


def _summary(path: str, sources: dict) -> dict:
    records = []
    for record in read_jsonl(path):
        source = sources.get(record.record_id)
        if source is None:
            continue
        source_question_ids = {
            str(question.get("id", question.get("question_id", "")))
            for question in source.get("questions", [])
        }
        qa_pairs = [
            qa
            for qa in record.qa_pairs
            if qa.question_id in source_question_ids or qa.is_generated_irrelevant
        ]
        records.append(record.model_copy(update={"qa_pairs": qa_pairs}))
    report = audit_records(records, sources)
    report["invalid_spans"] = sum(not atom.span_valid for record in records for atom in record.atoms)
    return report


def build_report(before: str, after: str, source: str) -> dict:
    sources = load_source_records(source)
    before_report = _summary(before, sources)
    after_report = _summary(after, sources)
    failures = [key for key in REQUIRED_ZERO if after_report.get(key) != 0]
    if after_report.get("answer_preservation_rate") != 1.0:
        failures.append("answer_preservation_rate")
    if after_report.get("invalid_spans") != 0:
        failures.append("invalid_spans")
    return {
        "before": before_report,
        "after": after_report,
        "accepted": not failures,
        "failed_gates": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = build_report(args.before, args.after, args.source)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from pathlib import Path

from doc_to_atom.atomization.models import StructuredRecord
from doc_to_atom.data.io import read_jsonl


def _tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).lower()
    return re.findall(r"\w+", normalized, flags=re.UNICODE)


def _grounding(answer: str, evidence: str) -> tuple[bool, float]:
    answer_tokens = _tokens(answer)
    evidence_tokens = _tokens(evidence)
    if not answer_tokens:
        return True, 1.0
    answer_text = " ".join(answer_tokens)
    evidence_text = " ".join(evidence_tokens)
    overlap = sum((Counter(answer_tokens) & Counter(evidence_tokens)).values())
    coverage = overlap / len(answer_tokens)
    return answer_text in evidence_text or coverage == 1.0, coverage


def audit_record(record: StructuredRecord) -> list[dict]:
    issues: list[dict] = []
    atom_by_id = {atom.atom_id: atom for atom in record.atoms}
    known_ids = set(atom_by_id)
    for qa in record.qa_pairs:
        base = {
            "record_id": record.record_id,
            "question_id": qa.question_id,
            "question": qa.question,
        }
        if not qa.is_irrelevant and not qa.gold_atom_ids:
            issues.append({**base, "severity": "error", "code": "answerable_empty_gold"})
        if qa.is_irrelevant and qa.gold_atom_ids:
            issues.append({**base, "severity": "error", "code": "irrelevant_has_gold"})

        roles = {
            "gold": qa.gold_atom_ids,
            "supporting": qa.supporting_atom_ids,
            "distractor": qa.distractor_atom_ids,
            "relevant": qa.relevant_atom_ids,
        }
        for role, values in roles.items():
            for atom_id in values:
                if atom_id not in known_ids:
                    issues.append(
                        {
                            **base,
                            "severity": "error",
                            "code": "unknown_atom_reference",
                            "role": role,
                            "atom_id": atom_id,
                        }
                    )

        expected_relevant = list(dict.fromkeys(qa.gold_atom_ids + qa.supporting_atom_ids))
        if qa.relevant_atom_ids != expected_relevant:
            issues.append(
                {
                    **base,
                    "severity": "error",
                    "code": "relevant_union_mismatch",
                    "expected": expected_relevant,
                    "actual": qa.relevant_atom_ids,
                }
            )

        if not qa.is_irrelevant and qa.gold_atom_ids and qa.answer:
            answers = qa.answer if isinstance(qa.answer, list) else [qa.answer]
            evidence = " ".join(
                f"{atom_by_id[atom_id].content} {atom_by_id[atom_id].source_span}"
                for atom_id in qa.gold_atom_ids
                if atom_id in atom_by_id
            )
            checks = [_grounding(str(answer), evidence) for answer in answers if str(answer)]
            if checks and not any(grounded for grounded, _coverage in checks):
                issues.append(
                    {
                        **base,
                        "severity": "warning",
                        "code": "weak_answer_grounding",
                        "answers": [str(answer) for answer in answers],
                        "max_token_coverage": max(coverage for _grounded, coverage in checks),
                        "gold_atom_ids": qa.gold_atom_ids,
                    }
                )
    return issues


def audit_records(records: list[StructuredRecord]) -> dict:
    issues = [issue for record in records for issue in audit_record(record)]
    return {
        "records": len(records),
        "questions": sum(len(record.qa_pairs) for record in records),
        "errors": sum(issue["severity"] == "error" for issue in issues),
        "warnings": sum(issue["severity"] == "warning" for issue in issues),
        "issues": issues,
    }


def audit_jsonl(path: str | Path) -> dict:
    return audit_records(read_jsonl(path))

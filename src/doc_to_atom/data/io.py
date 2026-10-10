from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from doc_to_atom.atomization.models import StructuredRecord
from doc_to_atom.atomization.validation import validate_record


def read_jsonl(path: str | Path) -> list[StructuredRecord]:
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    records.append(StructuredRecord.model_validate_json(line))
                except Exception as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return records


def write_jsonl(records: Iterable[StructuredRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(record.model_dump_json() + "\n")


def validate_jsonl(path: str | Path, reject_span_mismatch: bool = False) -> dict[str, int]:
    records = read_jsonl(path)
    atoms = questions = warnings = invalid_spans = 0
    for record in records:
        validate_record(record)
        atoms += len(record.atoms)
        questions += len(record.qa_pairs)
        warnings += len(record.annotation_meta.warnings)
        invalid_spans += sum(not atom.span_valid for atom in record.atoms)
    if reject_span_mismatch and invalid_spans:
        raise ValueError(f"{invalid_spans} atoms have source spans absent from their context")
    return {
        "records": len(records),
        "atoms": atoms,
        "questions": questions,
        "warnings": warnings,
        "invalid_spans": invalid_spans,
    }


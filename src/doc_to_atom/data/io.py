from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

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


def read_canonical_jsonl(path: str | Path) -> list[StructuredRecord]:
    """Consume the same deterministic representation as validation, without rewriting files."""
    return [validate_record(record) for record in read_jsonl(path)]


def write_jsonl(records: Iterable[StructuredRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(record.model_dump_json() + "\n")


def validate_jsonl(
    path: str | Path,
    reject_span_mismatch: bool = False,
    source_path: str | Path | None = None,
) -> dict:
    records = read_jsonl(path)
    atoms = questions = warnings = invalid_spans = 0
    invalid_span_details = []
    for record in records:
        validate_record(record)
        atoms += len(record.atoms)
        questions += len(record.qa_pairs)
        warnings += len(record.annotation_meta.warnings)
        invalid_spans += sum(not atom.span_valid for atom in record.atoms)
        invalid_span_details.extend(
            {"record_id": record.record_id, "atom_id": atom.atom_id,
             "source_span": atom.source_span, "context": record.context}
            for atom in record.atoms if not atom.span_valid
        )
    if reject_span_mismatch and invalid_spans:
        locations = ", ".join(f"{item['record_id']}/{item['atom_id']}" for item in invalid_span_details)
        raise ValueError(
            f"{invalid_spans} atoms have source spans absent from their context: {locations}. "
            "Run validation without --reject-span-mismatch to inspect invalid_span_details; "
            "no data has been modified."
        )
    from doc_to_atom.data.semantic_validation import audit_records, load_source_records

    sources = None if source_path is None else load_source_records(source_path)
    semantic = audit_records(records, sources)
    return {
        "records": len(records),
        "atoms": atoms,
        "questions": questions,
        "warnings": warnings,
        "invalid_spans": invalid_spans,
        "invalid_span_details": invalid_span_details,
        **{
            key: value
            for key, value in semantic.items()
            if key not in {"records", "questions", "warnings", "issues"}
        },
    }


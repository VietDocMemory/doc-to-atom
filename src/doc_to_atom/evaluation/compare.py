"""Compare prediction files only when their data and evaluation protocol agree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from doc_to_atom.evaluation.cli import score_rows


_SHARED_FIELDS = (
    "source_fingerprint", "atomized_fingerprint", "generation_protocol",
    "scoring_protocol", "evaluation_settings",
)
_SOURCE_FIELDS = ("question", "gold", "is_irrelevant", "split")


def compare_prediction_files(inputs: dict[str, str | Path]) -> dict:
    if len(inputs) < 2:
        raise ValueError("comparison requires at least two named prediction files")
    rows_by_name: dict[str, dict[str, dict]] = {}
    for name, path in inputs.items():
        rows = [
            json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        by_id: dict[str, dict] = {}
        for row in rows:
            question_id = row.get("question_id")
            if not question_id or question_id in by_id:
                raise ValueError(f"missing or duplicate question_id in {name}: {question_id}")
            by_id[question_id] = row
        rows_by_name[name] = by_id
    reference_name = next(iter(rows_by_name))
    reference = rows_by_name[reference_name]
    reasons: list[str] = []
    for name, by_id in rows_by_name.items():
        if by_id.keys() != reference.keys():
            raise ValueError(f"question IDs differ between {reference_name} and {name}")
        for question_id, row in by_id.items():
            baseline = reference[question_id]
            if any(row.get(field) != baseline.get(field) for field in _SOURCE_FIELDS):
                raise ValueError(f"source question/gold differs for {question_id}")
            for field in _SHARED_FIELDS:
                left, right = baseline.get(field), row.get(field)
                if left is None or right is None:
                    reason = f"legacy metadata missing: {field}"
                    if reason not in reasons:
                        reasons.append(reason)
                elif left != right:
                    raise ValueError(f"{field} mismatch for {question_id}")
    return {
        "comparable": not reasons,
        "reasons": reasons,
        "models": {
            name: score_rows(list(by_id.values())) for name, by_id in rows_by_name.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare prediction files under one protocol")
    parser.add_argument("--input", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    inputs: dict[str, Path] = {}
    for item in args.input:
        if "=" not in item:
            parser.error("--input must be NAME=PATH")
        name, path = item.split("=", 1)
        if not name or not path or name in inputs:
            parser.error("--input names and paths must be nonempty and names unique")
        inputs[name] = Path(path)
    report = compare_prediction_files(inputs)
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite comparison report: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))

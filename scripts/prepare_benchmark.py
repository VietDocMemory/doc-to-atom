"""Convert a benchmark split into grouped raw records for offline atomization.

Use ``--source-jsonl`` for a pinned local export, or ``--hf-dataset`` with an
explicit revision/config when reproducing a public Hugging Face dataset.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

from datasets import load_dataset

from doc_to_atom.data.preprocessors import preprocess_dataset_row


def _rows_from_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _preprocessor_name(dataset: str, override: str | None) -> str:
    if override:
        return override
    return "longbench" if dataset.startswith("longbench_") else dataset


def prepare_rows(
    rows: list[dict[str, Any]], dataset: str, split: str, preprocessor: str
) -> list[dict[str, Any]]:
    """Group questions by identical context, preserving source order."""
    grouped: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for index, row in enumerate(rows):
        item = preprocess_dataset_row(preprocessor, row)
        context = str(item["context"]).strip()
        question = str(item["question"]).strip()
        answers = item["answers"]
        if not context or not question or not answers:
            continue
        grouped.setdefault(context, []).append(
            {
                "id": str(row.get("id", row.get("_id", f"{dataset}_{split}_{index}"))),
                "question": question,
                "answer": answers,
                "question_type": "multi_hop" if dataset == "2wikimultihopqa" else "single_hop",
            }
        )
    return [
        {
            "id": f"{dataset}_{split}_{index}",
            "dataset": dataset,
            "split": split,
            "context": context,
            "questions": questions,
        }
        for index, (context, questions) in enumerate(grouped.items())
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a pinned paper benchmark split for atomization")
    parser.add_argument("--dataset", required=True, help="Output dataset label, e.g. squad or longbench_qasper")
    parser.add_argument("--split", required=True, choices=["train", "validation", "test"])
    parser.add_argument("--output", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-jsonl", help="Local raw benchmark export")
    source.add_argument("--hf-dataset", help="Hugging Face dataset identifier")
    parser.add_argument("--hf-config")
    parser.add_argument("--hf-split", help="Remote split name; defaults to --split")
    parser.add_argument("--hf-revision", help="Pin a commit hash for reproducibility")
    parser.add_argument("--preprocessor", help="Override parser: squad/drop/ropes/qasper/2wikimultihopqa/longbench")
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args()

    if args.source_jsonl:
        rows = _rows_from_jsonl(Path(args.source_jsonl))
    else:
        remote = load_dataset(
            args.hf_dataset,
            args.hf_config,
            split=args.hf_split or args.split,
            revision=args.hf_revision,
        )
        rows = [dict(row) for row in remote]
    if args.max_rows is not None:
        rows = rows[: args.max_rows]
    records = prepare_rows(rows, args.dataset, args.split, _preprocessor_name(args.dataset, args.preprocessor))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    print(json.dumps({"dataset": args.dataset, "split": args.split, "records": len(records)}, indent=2))


if __name__ == "__main__":
    main()

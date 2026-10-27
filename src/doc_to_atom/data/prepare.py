from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset


def group_squad_rows(rows, split: str) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["context"]].append(
            {
                "id": row["id"],
                "question": row["question"],
                "answers": list(row["answers"]["text"]),
                "question_type": "single_hop",
            }
        )
    return [
        {
            "id": f"squad_{split}_{index}",
            "dataset": "squad",
            "split": split,
            "context": context,
            "questions": questions,
        }
        for index, (context, questions) in enumerate(grouped.items())
    ]


def prepare_squad_subset(output: Path, train_rows: int = 16, validation_rows: int = 8) -> dict:
    """Prepare raw real-data inputs; atomization remains an offline MiniMax stage."""
    rows = []
    requested = (("train", train_rows), ("validation", validation_rows))
    for split, count in requested:
        if count == 0:
            continue
        dataset = load_dataset("rajpurkar/squad", split=f"{split}[:{count}]")
        rows.extend(group_squad_rows(dataset, split))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"records": len(rows), "train_source_rows": train_rows, "validation_source_rows": validation_rows}


def prepare_squad_records(
    output: Path, train_records: int, validation_records: int
) -> dict:
    """Stream distinct SQuAD contexts, counting JSONL records rather than QA rows.

    The streaming API does not need to materialize the full HF corpus in 32 GiB
    RAM. Question count per selected context depends on stream order; records
    are deliberately capped by unique context, not by original question count.
    """
    if train_records < 1 or validation_records < 1:
        raise ValueError("train_records and validation_records must be positive")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite source dataset: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = {}
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for split, limit in (("train", train_records), ("validation", validation_records)):
            seen = set()
            selected = 0
            for row in load_dataset("rajpurkar/squad", split=split, streaming=True):
                context = row["context"]
                if context in seen:
                    continue
                seen.add(context)
                record = group_squad_rows([row], split)[0]
                record["id"] = f"squad_{split}_{selected}"
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                selected += 1
                if selected == limit:
                    break
            if selected != limit:
                raise ValueError(f"SQuAD {split} contains only {selected} distinct contexts; requested {limit}")
            counts[split] = selected
    return {"records": sum(counts.values()), "train_records": counts["train"],
            "validation_records": counts["validation"],
            "questions_per_context": "first_observed_only"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["squad"], default="squad")
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-rows", type=int, default=16)
    parser.add_argument("--validation-rows", type=int, default=8)
    parser.add_argument("--train-records", type=int, help="Distinct-context JSONL rows (streaming)")
    parser.add_argument("--validation-records", type=int, help="Distinct-context JSONL rows (streaming)")
    args = parser.parse_args()
    if (args.train_records is None) != (args.validation_records is None):
        parser.error("--train-records and --validation-records must be provided together")
    result = (
        prepare_squad_records(Path(args.output), args.train_records, args.validation_records)
        if args.train_records is not None
        else prepare_squad_subset(Path(args.output), args.train_rows, args.validation_rows)
    )
    print(json.dumps(result, indent=2))


from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset


def prepare_squad_subset(output: Path, train_rows: int = 16, validation_rows: int = 8) -> dict:
    """Prepare raw real-data inputs; atomization remains an offline MiniMax stage."""
    train = load_dataset("rajpurkar/squad", split=f"train[:{train_rows}]")
    validation = load_dataset("rajpurkar/squad", split=f"validation[:{validation_rows}]")
    rows = []
    for split, dataset in (("train", train), ("validation", validation)):
        grouped = defaultdict(list)
        for row in dataset:
            grouped[row["context"]].append(
                {
                    "id": row["id"],
                    "question": row["question"],
                    "answer": row["answers"]["text"],
                    "question_type": "single_hop",
                }
            )
        for index, (context, questions) in enumerate(grouped.items()):
            rows.append(
                {
                    "id": f"squad_{split}_{index}",
                    "dataset": "squad",
                    "split": split,
                    "context": context,
                    "questions": questions,
                }
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"records": len(rows), "train_source_rows": train_rows, "validation_source_rows": validation_rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["squad"], default="squad")
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-rows", type=int, default=16)
    parser.add_argument("--validation-rows", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(prepare_squad_subset(Path(args.output), args.train_rows, args.validation_rows), indent=2))


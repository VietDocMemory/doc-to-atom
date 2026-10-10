from __future__ import annotations

import argparse
import json
from pathlib import Path

from doc_to_atom.data.io import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a balanced real-data overfit sanity slice")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--copies", type=int, default=5)
    args = parser.parse_args()

    candidates = []
    for record in read_jsonl(args.input):
        answerable = [
            qa
            for qa in record.qa_pairs
            if not qa.is_irrelevant and not qa.is_generated_irrelevant and qa.gold_atom_ids
        ]
        irrelevant = [qa for qa in record.qa_pairs if qa.is_generated_irrelevant]
        if record.split == "train" and answerable and irrelevant:
            candidates.append((len(record.atoms), record, answerable[0], irrelevant[0]))
    if not candidates:
        raise ValueError("no train record contains both a gold-mapped answerable and generated probe")

    _, source, answerable, irrelevant = min(candidates, key=lambda item: item[0])
    records = []
    for index in range(args.copies):
        clone = source.model_copy(deep=True)
        clone.record_id = f"{source.record_id}_overfit_train_{index}"
        clone.split = "train"
        clone.qa_pairs = [answerable.model_copy(deep=True), irrelevant.model_copy(deep=True)]
        records.append(clone)
    validation = source.model_copy(deep=True)
    validation.record_id = f"{source.record_id}_overfit_validation"
    validation.split = "validation"
    validation.qa_pairs = [answerable.model_copy(deep=True), irrelevant.model_copy(deep=True)]
    records.append(validation)
    write_jsonl(records, Path(args.output))
    print(
        json.dumps(
            {
                "source_record": source.record_id,
                "atoms": len(source.atoms),
                "train_copies": args.copies,
                "validation_copies": 1,
                "questions_per_record": 2,
                "answerable_question": answerable.question_id,
                "irrelevant_question": irrelevant.question_id,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

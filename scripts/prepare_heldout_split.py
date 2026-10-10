from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from doc_to_atom.data.io import read_jsonl, write_jsonl
from doc_to_atom.data.semantic_validation import audit_record


def _context_hash(context: str) -> str:
    return hashlib.sha256(context.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a document-disjoint held-out slice")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--train-docs", type=int, default=15)
    parser.add_argument("--validation-docs", type=int, default=3)
    parser.add_argument("--test-docs", type=int, default=3)
    args = parser.parse_args()

    records = read_jsonl(args.input)
    expected = args.train_docs + args.validation_docs + args.test_docs
    if len(records) < expected:
        raise ValueError(f"need {expected} records, found {len(records)}")
    hashes = [_context_hash(record.context) for record in records]
    if len(hashes) != len(set(hashes)):
        raise ValueError("source contains duplicate document contexts")

    eligible = []
    excluded = []
    for record in records:
        issues = audit_record(record)
        warned_questions = {issue["question_id"] for issue in issues}
        answerable = [
            qa
            for qa in record.qa_pairs
            if not qa.is_irrelevant and qa.gold_atom_ids and qa.question_id not in warned_questions
        ]
        irrelevant = [qa for qa in record.qa_pairs if qa.is_generated_irrelevant]
        if answerable and irrelevant:
            eligible.append((record, answerable, irrelevant))
        else:
            excluded.append(
                {
                    "record_id": record.record_id,
                    "reason": "lacks_clean_balanced_qa_pair",
                    "answerable_candidates": len(answerable),
                    "irrelevant_candidates": len(irrelevant),
                }
            )
    if len(eligible) < expected:
        raise ValueError(f"need {expected} eligible records, found {len(eligible)}")

    rng = random.Random(args.seed)
    rng.shuffle(eligible)
    splits = (
        ["train"] * args.train_docs
        + ["validation"] * args.validation_docs
        + ["test"] * args.test_docs
    )
    selected = []
    manifest_records = []
    for (record, answerable, irrelevant), split in zip(eligible[:expected], splits, strict=True):
        clone = record.model_copy(deep=True)
        clone.split = split
        clone.qa_pairs = [answerable[0].model_copy(deep=True), irrelevant[0].model_copy(deep=True)]
        selected.append(clone)
        manifest_records.append(
            {
                "record_id": clone.record_id,
                "split": split,
                "context_sha256": _context_hash(clone.context),
                "question_ids": [qa.question_id for qa in clone.qa_pairs],
            }
        )

    split_hashes = {
        split: {item["context_sha256"] for item in manifest_records if item["split"] == split}
        for split in ("train", "validation", "test")
    }
    if any(
        split_hashes[left] & split_hashes[right]
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise AssertionError("document leakage detected")

    write_jsonl(selected, args.output)
    Path(args.manifest).write_text(
        json.dumps(
            {
                "seed": args.seed,
                "document_disjoint": True,
                "counts": {
                    split: sum(item["split"] == split for item in manifest_records)
                    for split in ("train", "validation", "test")
                },
                "excluded": excluded,
                "records": manifest_records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"records": len(selected), "document_disjoint": True}, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

from doc_to_atom.data.io import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply reviewed, deterministic SQuAD mapping fixes")
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    path = Path(args.input)
    records = read_jsonl(path)
    matched = 0
    for record in records:
        if record.record_id != "squad_validation_0":
            continue
        atom = next((item for item in record.atoms if item.atom_id == "atom_1"), None)
        if atom is None or "Super Bowl" not in atom.content or "NFL" not in atom.content:
            raise ValueError("reviewed atom_1 precondition no longer holds")
        for qa in record.qa_pairs:
            if qa.question_id == "56d9895ddc89441400fdb50e":
                if qa.question != "The name of the NFL championship game is?":
                    raise ValueError("reviewed question precondition no longer holds")
                if qa.answer != "Super Bowl" or qa.is_irrelevant:
                    raise ValueError("reviewed answer/relevance precondition no longer holds")
                qa.gold_atom_ids = ["atom_1"]
                qa.relevant_atom_ids = ["atom_1", *qa.supporting_atom_ids]
                matched += 1
    if matched != 1:
        raise ValueError(f"expected exactly one reviewed mapping, found {matched}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_jsonl(records, temporary)
    temporary.replace(path)
    print("fixed squad_validation_0/56d9895ddc89441400fdb50e -> atom_1")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .client import MiniMaxAtomizer, MiniMaxConfig
from .models import QuestionAnnotation


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline MiniMax-M2.5 semantic atomization")
    parser.add_argument("--input", required=True, help="JSONL with id, context, questions")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    atomizer = MiniMaxAtomizer(MiniMaxConfig.from_env())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with Path(args.input).open(encoding="utf-8") as source, output.open("w", encoding="utf-8") as sink:
        for index, line in enumerate(source):
            raw = json.loads(line)
            questions = [
                QuestionAnnotation(
                    question_id=str(item.get("id", f"q_{q_index}")),
                    question=item["question"],
                    answer=item.get("answer"),
                    question_type=item.get("question_type", "single_hop"),
                )
                for q_index, item in enumerate(raw.get("questions", []))
            ]
            record = atomizer.atomize(
                str(raw.get("id", index)),
                raw["context"],
                questions,
                raw.get("dataset"),
                raw.get("split"),
            )
            if record is not None:
                sink.write(record.model_dump_json() + "\n")


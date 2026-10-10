from __future__ import annotations

import argparse
import json
from pathlib import Path

from .metrics import refusal_f1, rouge_l, word_f1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.predictions).read_text(encoding="utf-8").splitlines() if line]
    for row in rows:
        gold = row["gold"][0] if isinstance(row["gold"], list) else row["gold"]
        row["f1"] = word_f1(row["prediction"], gold)
        row["rouge_l"] = rouge_l(row["prediction"], gold)
        if row.get("is_irrelevant"):
            row["refusal_f1"] = refusal_f1(row["prediction"])
    summary = {key: sum(row.get(key, 0) for row in rows) / max(len(rows), 1) for key in ("f1", "rouge_l", "refusal_f1")}
    print(json.dumps(summary, indent=2))


from __future__ import annotations

import argparse
import json

from doc_to_atom.evaluation.benchmark import read_prediction_jsonl, write_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Score and tabulate paper-style benchmark predictions")
    parser.add_argument("--predictions", action="append", required=True, help="One or more JSONL files")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    rows = [row for path in args.predictions for row in read_prediction_jsonl(path)]
    print(json.dumps(write_report(rows, args.output_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

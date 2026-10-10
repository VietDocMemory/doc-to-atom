from __future__ import annotations

import argparse
import json

from .io import validate_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--reject-span-mismatch", action="store_true")
    args = parser.parse_args()
    print(json.dumps(validate_jsonl(args.input, args.reject_span_mismatch), indent=2))


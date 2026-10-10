from __future__ import annotations

import argparse
import json

from doc_to_atom.data.semantic_validation import audit_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Report semantic annotation integrity")
    parser.add_argument("--input", required=True)
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()
    report = audit_jsonl(args.input)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.fail_on_error and report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

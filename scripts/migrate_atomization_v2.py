from __future__ import annotations

import argparse
import json

from doc_to_atom.atomization.migration import migrate_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate legacy atomization with source-owned answers")
    parser.add_argument("--legacy", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(migrate_jsonl(args.legacy, args.source, args.output), indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json

from .migration import migrate_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate legacy atomization to schema v2")
    parser.add_argument("--legacy", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(migrate_jsonl(args.legacy, args.source, args.output), indent=2))

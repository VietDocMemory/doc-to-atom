"""Prepare named D2A benchmark splits; atomization remains an explicit step."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset

from doc_to_atom.data.benchmark_catalog import get_benchmark_dataset
from prepare_benchmark import prepare_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare named D2A benchmark datasets for atomization")
    parser.add_argument("--datasets", nargs="+", required=True, help="Aliases, e.g. squad drop longbench_qasper")
    parser.add_argument("--output-dir", required=True, help="Directory for <dataset>_<split>.jsonl")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--max-rows", type=int, help="Optional cap per dataset for a smoke benchmark")
    parser.add_argument("--hf-revision", help="Optional commit hash applied to every Hugging Face source")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    for name in args.datasets:
        spec = get_benchmark_dataset(name)
        remote = load_dataset(spec.hf_dataset, spec.hf_config, split=spec.hf_split, revision=args.hf_revision)
        rows = [dict(row) for row in remote]
        if args.max_rows is not None:
            rows = rows[: args.max_rows]
        records = prepare_rows(rows, spec.name, args.split, spec.preprocessor)
        target = output_dir / f"{spec.name}_{args.split}.jsonl"
        target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
        manifest.append({"dataset": spec.name, "source": spec.hf_dataset, "config": spec.hf_config,
                         "source_split": spec.hf_split, "output": str(target), "records": len(records)})
    (output_dir / "prepare_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

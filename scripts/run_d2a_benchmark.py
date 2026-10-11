"""D2A-only equivalent of Doc-to-LoRA's ``run_eval.py --datasets ...``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from doc_to_atom.config import load_config, load_project_env, write_run_manifest
from doc_to_atom.data.benchmark_catalog import get_benchmark_dataset
from doc_to_atom.data.io import read_jsonl, validate_jsonl
from doc_to_atom.evaluation.benchmark import write_report
from doc_to_atom.training.checkpoint import load_checkpoint
from doc_to_atom.training.factory import build_system

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_heldout_generalization import _evaluate_base, _evaluate_doc2atom


def load_named_records(
    data_dir: str | Path, datasets: list[str], split: str, allow_invalid_spans: bool = False
):
    """Read each atomized <dataset>_<split>.jsonl after strict validation."""
    base = Path(data_dir)
    records, sources = [], []
    for name in datasets:
        spec = get_benchmark_dataset(name)
        path = base / f"{spec.name}_{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"missing atomized benchmark file: {path}. Run atomize.py after prepare_d2a_benchmark.py."
            )
        stats = validate_jsonl(path, reject_span_mismatch=not allow_invalid_spans)
        selected = [record for record in read_jsonl(path) if record.split == split]
        if not selected:
            raise ValueError(f"{path} contains no records with split='{split}'")
        records.extend(selected)
        sources.append({"dataset": spec.name, "path": str(path), **stats})
    return records, sources


def main() -> None:
    parser = argparse.ArgumentParser(description="Run D2A-only benchmark datasets in one command")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--datasets", nargs="+", required=True, help="Same aliases as prepare_d2a_benchmark.py")
    parser.add_argument("--data-dir", required=True, help="Directory holding atomized <dataset>_<split>.jsonl")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--allow-invalid-spans",
        action="store_true",
        help=(
            "Run despite atom source-span mismatches and record them in benchmark_run.json. "
            "Use only for diagnostic/smoke runs; strict validation is the default."
        ),
    )
    args = parser.parse_args()

    load_project_env()
    config = load_config(args.config)
    records, data = load_named_records(
        args.data_dir, args.datasets, args.split, allow_invalid_spans=args.allow_invalid_spans
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_run_manifest(config, output / "run_manifest.json")
    system, tokenizer, _targets = build_system(config)
    step = load_checkpoint({"encoder_projection": system.encoder.projection, "compiler": system.compiler,
                            "router": system.router}, args.checkpoint)
    system.eval()
    base_rows, _ = _evaluate_base(system, tokenizer, records, "base_context", config["evaluation"]["max_new_tokens"])
    atom_rows, _ = _evaluate_doc2atom(system, tokenizer, records, config)
    dataset_by_record = {record.record_id: record.dataset or "unknown" for record in records}
    rows = []
    for row in [*base_rows, *atom_rows]:
        result = dict(row)
        result.update({"model": args.model_label, "dataset": dataset_by_record[result["record_id"]],
                       "gold": result["target"], "split": args.split})
        rows.append(result)
    result = write_report(rows, output)
    result.update({"checkpoint": str(Path(args.checkpoint).resolve()), "checkpoint_step": step,
                   "datasets": data, "methods_present": ["base_context", "doc2atom"],
                   "span_validation": "diagnostic_allowed" if args.allow_invalid_spans else "strict"})
    (output / "benchmark_run.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

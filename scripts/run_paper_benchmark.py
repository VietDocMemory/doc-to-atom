"""Evaluate a trained Doc2Atom checkpoint and assemble a paper-style report.

Doc-to-LoRA is intentionally accepted as externally generated JSONL because the
official upstream implementation/checkpoints were unavailable during the audit.
Each external row needs ``dataset``, ``split``, ``prediction`` and ``gold``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from doc_to_atom.config import load_config, load_project_env, write_run_manifest
from doc_to_atom.data.io import read_jsonl, validate_jsonl
from doc_to_atom.evaluation.benchmark import (
    PAPER_METHOD_ORDER,
    read_prediction_jsonl,
    write_report,
)
from doc_to_atom.training.checkpoint import load_checkpoint
from doc_to_atom.training.factory import build_system

# Reuse the tested generation paths used by the held-out diagnostic.  They run
# greedy Base+ctx and Doc2Atom generation and attach retrieval diagnostics.
from run_heldout_generalization import _evaluate_base, _evaluate_doc2atom


def _parse_external(value: str) -> tuple[str, Path]:
    method, separator, path = value.partition("=")
    if not separator or method not in {"d2l_pt", "d2l_raw"} or not path:
        raise ValueError("--external-predictions must be d2l_pt=PATH or d2l_raw=PATH")
    return method, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paper-style test evaluation from a Doc2Atom checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True, help="Atomized JSONL containing test records")
    parser.add_argument("--checkpoint", required=True, help="Doc2Atom best.pt or last.pt")
    parser.add_argument("--model-label", required=True, help="Column label, e.g. gemma2_2b_it")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--external-predictions",
        action="append",
        default=[],
        help="Optional D2L predictions: d2l_pt=PATH or d2l_raw=PATH",
    )
    args = parser.parse_args()
    load_project_env()
    config = load_config(args.config)
    data_stats = validate_jsonl(args.data, reject_span_mismatch=True)
    records = [record for record in read_jsonl(args.data) if record.split == "test"]
    if not records:
        raise ValueError("benchmark data must include records with split='test'")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_run_manifest(config, output / "run_manifest.json")
    system, tokenizer, _targets = build_system(config)
    step = load_checkpoint(
        {
            "encoder_projection": system.encoder.projection,
            "compiler": system.compiler,
            "router": system.router,
        },
        args.checkpoint,
    )
    system.eval()
    base_rows, _ = _evaluate_base(
        system, tokenizer, records, "base_context", config["evaluation"]["max_new_tokens"]
    )
    atom_rows, _ = _evaluate_doc2atom(system, tokenizer, records, config)
    dataset_by_record = {record.record_id: record.dataset or "unknown" for record in records}
    rows = []
    for row in [*base_rows, *atom_rows]:
        row = dict(row)
        row.update(
            {
                "model": args.model_label,
                "dataset": dataset_by_record[row["record_id"]],
                "gold": row["target"],
                "split": "test",
            }
        )
        rows.append(row)
    for raw in args.external_predictions:
        method, path = _parse_external(raw)
        for row in read_prediction_jsonl(path):
            row = dict(row)
            row.setdefault("model", args.model_label)
            row["method"] = method
            rows.append(row)
    result = write_report(rows, output)
    result.update(
        {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_step": step,
            "data": data_stats,
            "methods_present": sorted({row["method"] for row in rows}),
            "missing_paper_methods": [
                method for method in PAPER_METHOD_ORDER if method not in {row["method"] for row in rows}
            ],
        }
    )
    (output / "benchmark_run.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

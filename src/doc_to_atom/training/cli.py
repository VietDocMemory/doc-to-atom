from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from doc_to_atom.config import (
    ROOT,
    load_config,
    load_project_env,
    require_disclosed_training_values,
    write_run_manifest,
)
from doc_to_atom.data.io import read_jsonl, validate_jsonl
from doc_to_atom.training.factory import build_system, parameter_counts
from doc_to_atom.training.loop import run_training


def main() -> None:
    load_project_env()
    parser = argparse.ArgumentParser(description="Doc2Atom teacher-student training entry point")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", default=str(ROOT / "outputs/train"))
    parser.add_argument("--no-reranker", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--max-steps", type=int, help="Smoke-only record-step limit")
    parser.add_argument("--max-questions", type=int, help="Smoke-only questions per record")
    parser.add_argument("--wandb-project", help="Enable optional Weights & Biases experiment tracking")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    config = load_config(args.config)
    if args.max_steps is not None:
        config["training"]["max_steps"] = args.max_steps
    if args.max_questions is not None:
        config["data"]["max_questions_per_sample"] = args.max_questions
    require_disclosed_training_values(config)
    stats = validate_jsonl(args.data, reject_span_mismatch=True)
    output = Path(args.output_dir)
    write_run_manifest(config, output / "run_manifest.json")
    if args.validate_only:
        print(json.dumps(stats, indent=2))
        return
    system, _tokenizer, targets = build_system(config, enable_reranker=not args.no_reranker)
    report = {"data": stats, "parameters": parameter_counts(system), "target_count": len(targets)}
    (output / "initialization_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    wandb_run = None
    metric_logger = None
    if args.wandb_project:
        from doc_to_atom.training.tracking import start_wandb

        wandb_run, metric_logger = start_wandb(
            project=args.wandb_project,
            entity=args.wandb_entity,
            run_name=args.wandb_run_name,
            mode=args.wandb_mode,
            config=config,
        )
    try:
        result = run_training(
            system, _tokenizer, read_jsonl(args.data), config, output, metric_logger
        )
        if wandb_run is not None:
            wandb_run.summary.update(result)
    finally:
        if wandb_run is not None:
            wandb_run.finish()
    print(json.dumps({**report, "training": result}, indent=2))

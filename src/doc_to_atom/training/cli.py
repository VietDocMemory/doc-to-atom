from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Callable

import yaml

from doc_to_atom.config import ROOT, load_and_resolve_config, load_project_env, validate_config
from doc_to_atom.data.io import read_jsonl, validate_jsonl
from doc_to_atom.runtime import RunContext, RunTracker
from doc_to_atom.runtime.run import atomic_write_json, atomic_write_yaml
from doc_to_atom.training.factory import build_system, parameter_counts
from doc_to_atom.training.loop import run_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Doc2Atom teacher-student training entry point")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument(
        "--output", "--output-dir", dest="output_dir", default=str(ROOT / "outputs/train")
    )
    parser.add_argument(
        "--resume",
        metavar="PATH|latest",
        help="Resume exact trainer state from a checkpoint directory or the run's last pointer",
    )
    parser.add_argument("--no-reranker", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--max-steps", type=int, help="Diagnostic optimizer-step limit")
    parser.add_argument("--max-questions", type=int, help="Diagnostic questions per record")
    parser.add_argument(
        "--wandb-mode", choices=["online", "offline", "disabled"], default=None
    )
    parser.add_argument("--wandb-project", default="doc-to-atom")
    parser.add_argument("--wandb-entity")
    return parser


def resolve_resume_argument(args: argparse.Namespace) -> Path | None:
    if args.resume is None:
        return None
    if args.resume == "latest":
        candidate = Path(args.output_dir).resolve() / "checkpoints" / "last.json"
    else:
        candidate = Path(args.resume).resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"resume checkpoint does not exist: {candidate}")
    return candidate


def execute(
    args: argparse.Namespace,
    *,
    system_factory: Callable = build_system,
    training_runner: Callable = run_training,
) -> dict:
    config_path = Path(args.config).resolve()
    config = load_and_resolve_config(config_path)
    if args.max_steps is not None:
        config["training"]["max_steps"] = args.max_steps
    if args.max_questions is not None:
        config["data"]["max_questions_per_sample"] = args.max_questions
    runtime = config.setdefault("runtime", {})
    if args.wandb_mode is not None:
        runtime["wandb_mode"] = args.wandb_mode
    validate_config(config, require_runnable=True)

    resume_path = resolve_resume_argument(args)
    output = Path(args.output_dir).resolve()
    context = RunContext.create(
        output,
        config=config,
        command=sys.argv,
        resume=resume_path is not None,
    )
    if resume_path is None:
        atomic_write_yaml(
            output / "config" / "requested.yaml",
            yaml.safe_load(config_path.read_text(encoding="utf-8")),
        )
    tracker = RunTracker.create(
        context,
        mode=runtime.get("wandb_mode", "offline"),
        project=args.wandb_project,
        entity=args.wandb_entity,
    )
    try:
        tracker.log_config()
        tracker.log_manifest(context.run_manifest_path)
        stats = validate_jsonl(args.data, reject_span_mismatch=True)
        atomic_write_json(output / "manifests" / "data_validation.json", stats)
        tracker.log_manifest(output / "manifests" / "data_validation.json")
        if args.validate_only:
            result = {"data": stats, "validated_only": True}
            atomic_write_json(output / "reports" / "validation.json", result)
            return result

        system, tokenizer, targets = system_factory(
            config, enable_reranker=not args.no_reranker
        )
        report = {
            "data": stats,
            "parameters": parameter_counts(system),
            "target_count": len(targets),
        }
        initialization_path = output / "reports" / "initialization.json"
        atomic_write_json(initialization_path, report)
        tracker.log_file(initialization_path)
        training_result = training_runner(
            system,
            tokenizer,
            read_jsonl(args.data),
            config,
            output,
            resume_from=resume_path,
            tracker=tracker,
        )
        result = {**report, "training": training_result}
        result_path = output / "reports" / "training_result.json"
        atomic_write_json(result_path, result)
        tracker.log_file(result_path)
        return result
    finally:
        tracker.finish()


def main() -> None:
    load_project_env()
    logging.basicConfig(level=logging.INFO)
    result = execute(build_parser().parse_args())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

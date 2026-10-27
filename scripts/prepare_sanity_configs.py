"""Create short, explicitly marked curriculum-boundary diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from doc_to_atom.config import load_and_resolve_config


BOUNDARIES = (("stage1", 0, 1), ("stage2", 2000, 4),
              ("stage3", 6000, 8), ("stage4", 9000, 8))


def prepare(base: Path, output: Path, steps: int = 2) -> list[Path]:
    if steps < 1:
        raise ValueError("steps must be positive")
    source = load_and_resolve_config(base, require_runnable=True)
    output.mkdir(parents=True, exist_ok=True)
    destinations = []
    for stage, start, top_k in BOUNDARIES:
        # Fully resolved snapshots avoid changing the source config and record
        # exactly which reconstruction assumptions are active in each run.
        config = yaml.safe_load(yaml.safe_dump(source))
        config["profile"] = {"name": f"{source['profile']['name']}-{stage}-sanity",
                             "family": "experiments"}
        config["deviations"] = [*source.get("deviations", []), {
            "id": f"CURRICULUM-DIAGNOSTIC-{stage.upper()}",
            "description": f"Begin at optimizer curriculum step {start} (K={top_k}); not sequential training",
            "rationale": "Exercise the boundary on a small annotated sample without 9000 updates",
            "affected_metrics": ["training_loss", "routing", "adapter_norm"],
        }]
        config["training"].update({
            "initial_curriculum_step": start,
            "max_steps": steps,
            "max_epochs": 1,
            "warmup_steps": 0,
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "effective_global_batch": 1,
            "checkpoint_every_steps": steps,
        })
        config["data"].update({"max_context_tokens": 1024, "max_answer_tokens": 64,
                               "max_questions_per_sample": 2})
        config.setdefault("runtime", {})["reranker_device"] = "cpu"
        path = output / f"{stage}.yaml"
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        destinations.append(path)
    return destinations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()
    for path in prepare(Path(args.base), Path(args.output), args.steps):
        print(path)


if __name__ == "__main__":
    main()

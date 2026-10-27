"""Prepare one continuous, shortened four-stage curriculum for a local experiment."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import yaml

from doc_to_atom.config import load_and_resolve_config


def prepare(base: Path, output: Path, *, stage_steps: int, train_records: int) -> dict:
    if not 200 <= stage_steps <= 500:
        raise ValueError("stage_steps must be within 200..500")
    if not 500 <= train_records <= 2000:
        raise ValueError("train_records must be within 500..2000")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    config = load_and_resolve_config(base, require_runnable=True)
    config["profile"] = {"name": "gemma-local-medium-shortened-curriculum", "family": "experiments"}
    config["deviations"] = [*config.get("deviations", []), {
        "id": "LOCAL-MEDIUM-CURRICULUM",
        "description": f"Continuous 4-stage curriculum with {stage_steps} optimizer steps per stage",
        "rationale": "Local diagnostic, not the paper's 2000/6000/9000 stage boundaries",
        "affected_metrics": ["losses", "routing", "answer_f1", "refusal_f1"],
    }]
    total_steps = 4 * stage_steps
    accumulation = 16
    config["training"].update({
        "curriculum_boundaries": [stage_steps, 2 * stage_steps, 3 * stage_steps],
        "max_steps": total_steps,
        "max_epochs": math.ceil(total_steps * accumulation / train_records) + 2,
        "initial_curriculum_step": 0,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": accumulation,
        "effective_global_batch": accumulation,
        "warmup_steps": min(100, stage_steps),
        "validation_every_steps": stage_steps,
        "checkpoint_every_steps": stage_steps,
        "stage4_only_best": True,
        # Do not stop at an early stage merely because a new loss was enabled.
        # This is an experiment-only choice, not an unpublished paper constant.
        "early_stopping_patience": max(100, math.ceil(total_steps * accumulation / train_records) + 3),
    })
    config["data"].update({
        "max_context_tokens": 1024,
        "max_atom_tokens": 96,
        "max_answer_tokens": 64,
        "max_questions_per_sample": 3,  # 1 source QA + 2 generated probes in this preparation mode.
        "max_atoms_per_sample": 64,
    })
    config.setdefault("runtime", {}).update({
        "distributed_backend": "none", "reranker_device": "cpu",
        "gradient_checkpointing": False, "checkpoint_upload": "none",
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage-steps", type=int, default=200)
    parser.add_argument("--train-records", type=int, required=True)
    args = parser.parse_args()
    config = prepare(Path(args.base), Path(args.output),
                     stage_steps=args.stage_steps, train_records=args.train_records)
    print(f"{args.output}: {config['training']['curriculum_boundaries']} "
          f"max_steps={config['training']['max_steps']}")


if __name__ == "__main__":
    main()

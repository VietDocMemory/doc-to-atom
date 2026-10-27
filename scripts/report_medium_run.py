"""Summarize observed medium-run stage logs and validation metrics; never invent results."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from doc_to_atom.config import load_config


STAGES = ("single_atom", "multi_atom", "robustness", "full_composition")


def summarize(run: Path, *, output: Path) -> dict:
    config_path = run / "config" / "resolved.yaml"
    log_path = run / "train_log.jsonl"
    if not config_path.is_file() or not log_path.is_file():
        raise FileNotFoundError("no completed training log/config; cannot report measured results")
    config = load_config(config_path)
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    expected_steps = int(config["training"]["max_steps"])
    result = {"run": str(run), "expected_steps": expected_steps,
              "completed_steps": rows[-1]["optimizer_step"] if rows else 0,
              "stages": {}, "nonfinite": [], "evaluation": None}
    for stage in STAGES:
        selected = [row for row in rows if row.get("curriculum_stage") == stage]
        losses = {}
        for name in ("lm", "distill", "route", "irrelevant", "protect", "sparse"):
            values = [row.get("train", {}).get("overall", {}).get("losses", {}).get(name)
                      for row in selected]
            values = [value for value in values if value is not None]
            if any(not math.isfinite(float(value)) for value in values):
                result["nonfinite"].append(f"{stage}/{name}")
            losses[name] = sum(values) / len(values) if values else None
        result["stages"][stage] = {
            "optimizer_updates": len(selected),
            "top_k": sorted({row.get("top_k") for row in selected if row.get("top_k") is not None}),
            "loss_means": losses,
        }
    metrics_path = run / "metrics" / "evaluation_validation.json"
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        result["evaluation"] = {
            key: metrics.get(key) for key in
            ("f1", "rouge_l", "refusal_f1", "latency_ms", "count",
             "answerable_count", "irrelevant_count", "peak_vram_bytes")
        }
    result["all_stages_observed"] = all(result["stages"][stage]["optimizer_updates"] > 0
                                     for stage in STAGES)
    result["training_completed"] = (
        result["completed_steps"] == expected_steps and result["all_stages_observed"]
        and not result["nonfinite"]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(Path(args.run), output=Path(args.output)), indent=2))


if __name__ == "__main__":
    main()

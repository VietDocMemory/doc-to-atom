"""Read-only audit of saved Molab run evidence. Does not load or execute model weights."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from doc_to_atom.config import load_config


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(run: Path) -> dict:
    config_path = run / "config/resolved.yaml"
    config = load_config(config_path)
    log_path = run / "train_log.jsonl"
    rows = [json.loads(s) for s in log_path.read_text(encoding="utf-8").splitlines() if s.strip()]
    validations = []
    for row in rows:
        if not row.get("validation"):
            continue
        report = row["validation"]
        validations.append({"step": row["optimizer_step"], "stage": row.get("curriculum_stage"),
            "overall_lm": report["overall"]["losses"]["lm"],
            "answerable_lm": report["answerable"]["losses"]["lm"],
            "irrelevant_lm": report["generated_irrelevant"]["losses"]["lm"],
            "answerable_count": report["answerable"]["count"],
            "irrelevant_count": report["generated_irrelevant"]["count"]})
    best = json.loads((run / "checkpoints/best.json").read_text(encoding="utf-8"))
    last = json.loads((run / "checkpoints/last.json").read_text(encoding="utf-8"))
    metrics_path = run / "metrics/evaluation_validation.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    stage4 = [r for r in validations if r["stage"] == "full_composition"]
    selected = next((r for r in validations if r["step"] == best["optimizer_step"]), None)
    nonfinite = [r["optimizer_step"] for r in rows if any(not math.isfinite(float(r[key]))
                 for key in ("learning_rate", "gradient_norm") if r.get(key) is not None)]
    warmup = config["training"]["warmup_steps"]
    maximum = config["training"].get("max_steps")
    old_protocol = config["training"].get("objective_protocol", "legacy_v1")
    inspected = [config_path, log_path, run / "checkpoints/best.json", run / "checkpoints/last.json"]
    states = {}
    for name in {best["path"], last["path"]}:
        state_path = run / "checkpoints" / name / "state.json"
        inspected.append(state_path)
        state = json.loads(state_path.read_text(encoding="utf-8"))
        states[name] = {key: state.get(key) for key in (
            "optimizer_step", "config_fingerprint", "dataset_fingerprint", "prompt_fingerprint")}
    findings = []
    if old_protocol == "legacy_v1":
        findings += [
            {"code": "TRAIN_INFERENCE_PROMPT_MISMATCH", "status": "code_confirmed",
             "detail": "Legacy training omits short-answer/refusal instructions used in generation."},
            {"code": "NO_SUPERVISED_ANSWER_TERMINATOR", "status": "code_confirmed",
             "detail": "Legacy target is source answer text only; generation stops on a completed answer line."},
            {"code": "SEQUENCE_VS_TOKEN_MEAN_KL", "status": "code_confirmed",
             "detail": "Legacy loop KL batchmean on [1,T,V] grows with T, unlike the masked token-mean helper."},
        ]
    findings += [
        {"code": "REFUSAL_HEAVY_VALIDATION", "status": "measured", "detail": selected},
        {"code": "SHORT_RUN_WARMUP", "status": "configured",
         "warmup_steps": warmup, "max_steps": maximum,
         "warmup_fraction": warmup / maximum if maximum else None},
        {"code": "TEACHER_FORCED_EXAMPLES_NOT_GENERATION", "status": "code_confirmed",
         "detail": "Training report predictions are tokenwise teacher-forced argmax, not free generation."},
        {"code": "CHECKPOINT_DATA_BINDING", "status": "metadata_observed", "detail": states},
    ]
    if validations and rows:
        last_train = rows[-1].get("train", {}).get("answerable", {}).get("losses", {}).get("lm")
        findings.append({"code": "SOURCE_TRAIN_VALIDATION_GAP", "status": "measured",
                         "last_train_answerable_lm": last_train,
                         "last_validation_answerable_lm": validations[-1]["answerable_lm"],
                         "note": "A large gap suggests poor generalization; more steps alone are not a demonstrated fix."})
    if metrics_path.exists():
        inspected.append(metrics_path)
    return {"run": str(run), "read_only": True,
        "completed_optimizer_steps": rows[-1]["optimizer_step"] if rows else 0,
        "stage_update_counts": dict(Counter(r.get("curriculum_stage") for r in rows)),
        "nonfinite_optimizer_steps": nonfinite, "best_pointer": best, "last_pointer": last,
        "selected_validation": selected,
        "stage4_best_by_answerable_lm": min(stage4, key=lambda r: r["answerable_lm"]) if stage4 else None,
        "validation_history": validations, "evaluation": metrics, "findings": findings,
        "evidence_fingerprints": {p.relative_to(run).as_posix(): digest(p) for p in inspected},
        "limitation": "Static/log audit: no A/B generation or retraining; causal F1 improvement is unmeasured."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.run.resolve()):
        raise ValueError("Write the audit outside the historical run directory")
    if args.output.exists():
        raise FileExistsError(args.output)
    report = audit(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "steps": report["completed_optimizer_steps"],
                      "findings": [item["code"] for item in report["findings"]]}, indent=2))


if __name__ == "__main__":
    main()

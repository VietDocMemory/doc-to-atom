from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch

from doc_to_atom.config import (
    config_fingerprint,
    config_fingerprint_candidates,
    legacy_config_fingerprint,
    load_config,
    load_project_env,
)
from doc_to_atom.data.io import read_jsonl, validate_jsonl
from doc_to_atom.evaluation.metrics import (
    exact_match,
    first_line,
    is_refusal,
    rouge_l,
    word_f1,
)
from doc_to_atom.inference.query import QueryEngine
from doc_to_atom.training.checkpoint import load_training_checkpoint
from doc_to_atom.training.factory import build_system
from doc_to_atom.training.loop import _memory


def routing_scores(
    ranked_ids: list[str], gold_ids: list[str], candidate_count: int,
    ks: tuple[int, ...] = (1, 4, 8),
) -> dict:
    if candidate_count < len(ranked_ids) or len(set(ranked_ids)) != len(ranked_ids):
        raise ValueError("ranked atom IDs are duplicated or exceed candidate count")
    gold = set(gold_ids)
    if not gold:
        return {**{f"recall_at_{k}": None for k in ks}, "mrr": None}
    first_rank = next((index for index, atom_id in enumerate(ranked_ids, 1) if atom_id in gold), None)
    result = {"mrr": 0.0 if first_rank is None else 1.0 / first_rank}
    for k in ks:
        result[f"recall_at_{k}"] = len(gold.intersection(ranked_ids[:k])) / len(gold)
        result[f"routing_saturation_at_{k}"] = (
            min(k, candidate_count) / candidate_count if candidate_count else None
        )
    return result


def enrich_rows(rows: list[dict], records: list) -> None:
    """Join legacy predictions to validated atomized questions without changing source gold."""
    if not rows:
        raise ValueError("cannot join empty prediction rows")
    splits = {row.get("split") for row in rows}
    source: dict[str, tuple] = {}
    for record in records:
        if record.split not in splits:
            continue
        for qa in record.qa_pairs:
            if qa.question_id in source:
                raise ValueError(f"duplicate source question_id: {qa.question_id}")
            source[qa.question_id] = (record, qa)
    seen: set[str] = set()
    for row in rows:
        question_id = row.get("question_id")
        if question_id in seen:
            raise ValueError(f"duplicate prediction question_id: {question_id}")
        seen.add(question_id)
        if question_id not in source:
            raise ValueError(f"unknown prediction question_id: {question_id}")
        record, qa = source[question_id]
        if (
            row.get("record_id") != record.record_id
            or row.get("split") != record.split
            or row.get("question") != qa.question
            or row.get("gold") != list(qa.answers)
            or row.get("is_irrelevant") is not qa.is_irrelevant
        ):
            raise ValueError(f"source fields differ for question_id: {question_id}")
        row["gold_atom_ids"] = list(qa.gold_atom_ids)
        row["atom_count"] = len(record.atoms)
        if "ranked_atom_ids" in row:
            atom_ids = {atom.atom_id for atom in record.atoms}
            if not set(row["ranked_atom_ids"]).issubset(atom_ids):
                raise ValueError(f"unknown ranked atom ID for question_id: {question_id}")
            row["candidate_count"] = len(record.atoms)
    missing = source.keys() - seen
    if missing:
        raise ValueError(f"missing prediction question_id: {sorted(missing)}")


def score_rows(rows: list[dict], *, refusal_literal: str = "unanswerable") -> dict:
    answerable, irrelevant, latencies, adapter_norms = [], [], [], []
    recall_values = {k: [] for k in (1, 4, 8)}
    saturation_values = {k: [] for k in (1, 4, 8)}
    ranks = []
    tp = fp = fn = tn = 0
    for row in rows:
        prediction = row["prediction"]
        refused = is_refusal(prediction, refusal_literal)
        row["predicted_refusal"] = refused
        if row.get("is_irrelevant"):
            row["refusal_f1"] = float(refused)
            irrelevant.append(row)
            if refused:
                tp += 1
            else:
                fn += 1
        else:
            gold = row["gold"]
            references = gold if isinstance(gold, list) else [gold]
            if not references:
                raise ValueError(f"answerable prediction lacks source gold: {row.get('question_id')}")
            row["f1"] = max(word_f1(prediction, answer) for answer in references)
            row["raw_f1"] = row["f1"]
            row["first_line_f1"] = max(
                word_f1(first_line(prediction), answer) for answer in references
            )
            row["exact_match"] = max(exact_match(prediction, answer) for answer in references)
            row["rouge_l"] = max(rouge_l(prediction, answer) for answer in references)
            answerable.append(row)
            if refused:
                fp += 1
            else:
                tn += 1
            if "gold_atom_ids" in row and "ranked_atom_ids" in row:
                route = routing_scores(
                    row["ranked_atom_ids"], row["gold_atom_ids"],
                    row.get("candidate_count", len(row["ranked_atom_ids"])),
                )
                row.update(route)
                row["recall_at_k"] = route["recall_at_8"]
                if route["mrr"] is not None:
                    ranks.append(route["mrr"])
                    for k in recall_values:
                        recall_values[k].append(route[f"recall_at_{k}"])
                        if route[f"routing_saturation_at_{k}"] is not None:
                            saturation_values[k].append(route[f"routing_saturation_at_{k}"])
        if row.get("latency_ms") is not None:
            latencies.append(float(row["latency_ms"]))
        if row.get("adapter_norm") is not None:
            adapter_norms.append(float(row["adapter_norm"]))

    mean = lambda values: statistics.fmean(values) if values else None
    precision = tp / (tp + fp) if tp + fp else (0.0 if irrelevant else None)
    recall = tp / len(irrelevant) if irrelevant else None
    binary_f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else (0.0 if recall is not None else None)
    )
    return {
        "scoring_protocol": "squad_v1_multiref",
        "count": len(rows),
        "answerable_count": len(answerable),
        "irrelevant_count": len(irrelevant),
        "f1": mean([row["f1"] for row in answerable]),
        "raw_f1": mean([row["raw_f1"] for row in answerable]),
        "first_line_f1": mean([row["first_line_f1"] for row in answerable]),
        "exact_match": mean([row["exact_match"] for row in answerable]),
        "rouge_l": mean([row["rouge_l"] for row in answerable]),
        "refusal_confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "refusal_precision": precision,
        "refusal_recall": recall,
        "refusal_f1": binary_f1,
        "false_refusal_rate": fp / len(answerable) if answerable else None,
        "false_answer_rate": fn / len(irrelevant) if irrelevant else None,
        "routing_evaluated_count": len(ranks),
        **{f"recall_at_{k}": mean(values) for k, values in recall_values.items()},
        **{
            f"routing_saturation_at_{k}": mean(values)
            for k, values in saturation_values.items()
        },
        "recall_at_k": mean(recall_values[8]),
        "mrr": mean(ranks),
        "latency_ms": mean(latencies),
        "adapter_norm": mean(adapter_norms),
        "no_memory_weight_answerable": mean([float(row["no_memory_weight"]) for row in answerable if row.get("no_memory_weight") is not None]),
        "no_memory_weight_irrelevant": mean([float(row["no_memory_weight"]) for row in irrelevant if row.get("no_memory_weight") is not None]),
    }


@torch.inference_mode()
def generate_rows(
    system, tokenizer, records, config, *, engine_factory=QueryEngine, memory_builder=_memory
) -> list[dict]:
    engine = engine_factory(
        system.base_model, tokenizer, system.encoder, system.router,
        system.composer, system.injector,
    )
    rows = []
    top_k = int(config.get("evaluation", {}).get("routing_recall_k", 8))
    for record in records:
        bank = memory_builder(system, tokenizer, record, config)
        for qa in record.qa_pairs:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            started = time.perf_counter()
            result = engine.answer(
                qa.question, bank, top_k=top_k,
                max_new_tokens=config["evaluation"]["max_new_tokens"],
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            ranked = result["ranked_atom_ids"]
            route = routing_scores(ranked, qa.gold_atom_ids, len(record.atoms))
            rows.append({
                "record_id": record.record_id,
                "split": record.split,
                "question_id": qa.question_id,
                "question": qa.question,
                "prediction": result["prediction"],
                "raw_generation": result.get("raw_generation"),
                "stop_reason": result.get("stop_reason"),
                "gold": list(qa.answers),
                "gold_atom_ids": list(qa.gold_atom_ids),
                "is_irrelevant": qa.is_irrelevant,
                "atom_count": len(record.atoms),
                "candidate_count": len(record.atoms),
                "selected_atom_ids": result["selected_atom_ids"],
                "ranked_atom_ids": ranked,
                "routing_weights": result["routing_weights"],
                "no_memory_weight": result.get("no_memory_weight"),
                "recall_at_k": route["recall_at_8"],
                **route,
                "latency_ms": (time.perf_counter() - started) * 1000,
                "adapter_norm": result["composed_adapter_norm"],
            })
    return rows


def _checkpoint_directory(path: Path) -> Path:
    if path.is_file():
        pointer = json.loads(path.read_text(encoding="utf-8"))
        path = path.parent / pointer["path"]
    if (path / "distributed.json").is_file():
        path /= "portable"
    return path


def main() -> None:
    load_project_env()
    parser = argparse.ArgumentParser(description="Generate and score checkpoint predictions")
    parser.add_argument("--predictions", help="Score an existing predictions JSONL")
    parser.add_argument("--config", help="Resolved training config snapshot")
    parser.add_argument("--data", help="Schema-v2 atomized JSONL")
    parser.add_argument("--source", help="Authoritative source JSONL")
    parser.add_argument("--checkpoint", help="Checkpoint directory or last/best pointer")
    parser.add_argument("--output", help="Predictions JSONL to create")
    parser.add_argument("--split", choices=["validation", "test", "all"], default="validation")
    parser.add_argument("--no-reranker", action="store_true")
    parser.add_argument("--wandb-mode", choices=["disabled", "offline", "online"], default="disabled")
    parser.add_argument("--wandb-project", default="doc-to-atom")
    args = parser.parse_args()

    if args.predictions:
        if any((args.config, args.checkpoint, args.output)):
            parser.error("--predictions cannot be combined with generation arguments")
        if bool(args.data) != bool(args.source):
            parser.error("score-only source join requires both --data and --source")
        rows = [
            json.loads(line)
            for line in Path(args.predictions).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if args.data:
            stats = validate_jsonl(args.data, source_path=args.source, reject_span_mismatch=True)
            if stats["errors"] or stats["invalid_spans"]:
                raise ValueError(f"atomized data failed source contract: {stats}")
            enrich_rows(rows, read_jsonl(args.data))
    else:
        if not all((args.config, args.data, args.source, args.checkpoint, args.output)):
            parser.error("generation requires --config --data --source --checkpoint --output")
        destination = Path(args.output)
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite predictions: {destination}")
        stats = validate_jsonl(args.data, source_path=args.source, reject_span_mismatch=True)
        if stats["errors"] or stats["invalid_spans"]:
            raise ValueError(f"atomized data failed source contract: {stats}")
        records = [
            record for record in read_jsonl(args.data)
            if args.split == "all" or record.split == args.split
        ]
        if not records:
            raise ValueError(f"no records for split {args.split}")
        config = load_config(args.config)
        expected_config_fingerprints = config_fingerprint_candidates(config)
        system, tokenizer, _targets = build_system(config, enable_reranker=not args.no_reranker)
        modules = {
            "encoder_projection": system.encoder.projection,
            "compiler": system.compiler,
            "router": system.router,
        }
        state = load_training_checkpoint(
            modules, _checkpoint_directory(Path(args.checkpoint)),
            expected_fingerprints={"config": expected_config_fingerprints}, restore_rng=False,
        )
        system.eval()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        rows = generate_rows(system, tokenizer, records, config)
        source_fingerprint = hashlib.sha256(Path(args.source).read_bytes()).hexdigest()
        atomized_fingerprint = hashlib.sha256(Path(args.data).read_bytes()).hexdigest()
        settings = {
            "split": args.split,
            "max_new_tokens": config["evaluation"]["max_new_tokens"],
            "routing_recall_k": config["evaluation"].get("routing_recall_k", 8),
            "reranker": not args.no_reranker,
            "refusal_literal": "unanswerable",
        }
        for row in rows:
            row["checkpoint_step"] = state.optimizer_step
            row["config_fingerprint"] = config_fingerprint(config)
            row["source_fingerprint"] = source_fingerprint
            row["atomized_fingerprint"] = atomized_fingerprint
            row["generation_protocol"] = "short_answer_v1"
            row["prompt_id"] = "doc2atom_qa_v1"
            row["scoring_protocol"] = "squad_v1_multiref"
            row["evaluation_settings"] = settings
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )

    summary = score_rows(rows)
    if args.predictions and rows:
        for key in (
            "checkpoint_step", "config_fingerprint", "source_fingerprint",
            "atomized_fingerprint", "generation_protocol", "evaluation_settings",
        ):
            values = [row.get(key) for row in rows]
            if any(value != values[0] for value in values[1:]):
                raise ValueError(f"inconsistent prediction provenance: {key}")
            if values[0] is not None:
                summary[key] = values[0]
    if not args.predictions:
        summary["checkpoint_step"] = state.optimizer_step
        summary["config_fingerprint"] = config_fingerprint(config)
        summary["source_fingerprint"] = source_fingerprint
        summary["atomized_fingerprint"] = atomized_fingerprint
        summary["generation_protocol"] = "short_answer_v1"
        summary["evaluation_settings"] = settings
        summary["peak_vram_bytes"] = (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        )
        output_root = destination.parent.parent if destination.parent.name == "predictions" else destination.parent
        metrics_path = output_root / "metrics" / f"evaluation_{args.split}.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        report_path = output_root / "reports" / f"evaluation_{args.split}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps({
            "predictions_path": str(destination),
            "metrics_path": str(metrics_path),
            "checkpoint_path": str(args.checkpoint),
            "checkpoint_step": state.optimizer_step,
            "settings": settings,
            "metrics": summary,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.wandb_mode != "disabled":
        import wandb

        provenance = {
            key: summary.get(key) for key in (
                "checkpoint_step", "config_fingerprint", "source_fingerprint",
                "atomized_fingerprint", "generation_protocol", "scoring_protocol",
                "evaluation_settings",
            ) if summary.get(key) is not None
        }
        run = wandb.init(
            project=args.wandb_project, mode=args.wandb_mode, config=provenance,
        )
        try:
            run.log({f"eval/{key}": value for key, value in summary.items() if isinstance(value, (int, float)) and value is not None})
            run.save(str(Path(args.predictions) if args.predictions else destination))
            if not args.predictions:
                run.save(str(metrics_path))
                run.save(str(report_path))
                run.save(str(Path(args.config)))
                manifest = output_root / "manifests" / "run.json"
                if manifest.is_file():
                    run.save(str(manifest))
        finally:
            run.finish()
    print(json.dumps(summary, ensure_ascii=False, indent=2))

from __future__ import annotations

import argparse
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
from doc_to_atom.evaluation.metrics import refusal_f1, rouge_l, word_f1
from doc_to_atom.inference.query import QueryEngine
from doc_to_atom.training.checkpoint import load_training_checkpoint
from doc_to_atom.training.factory import build_system
from doc_to_atom.training.loop import _memory


def score_rows(rows: list[dict]) -> dict:
    answerable, irrelevant, recalls, ranks, latencies, adapter_norms = [], [], [], [], [], []
    for row in rows:
        prediction = row["prediction"]
        if row.get("is_irrelevant"):
            row["refusal_f1"] = refusal_f1(prediction)
            irrelevant.append(row)
        else:
            gold = row["gold"]
            references = gold if isinstance(gold, list) else [gold]
            if not references:
                raise ValueError(f"answerable prediction lacks source gold: {row.get('question_id')}")
            row["f1"] = max(word_f1(prediction, answer) for answer in references)
            row["rouge_l"] = max(rouge_l(prediction, answer) for answer in references)
            answerable.append(row)
            if row.get("recall_at_k") is not None:
                recalls.append(float(row["recall_at_k"]))
            if row.get("mrr") is not None:
                ranks.append(float(row["mrr"]))
        if row.get("latency_ms") is not None:
            latencies.append(float(row["latency_ms"]))
        if row.get("adapter_norm") is not None:
            adapter_norms.append(float(row["adapter_norm"]))

    mean = lambda values: statistics.fmean(values) if values else None
    return {
        "count": len(rows),
        "answerable_count": len(answerable),
        "irrelevant_count": len(irrelevant),
        "f1": mean([row["f1"] for row in answerable]),
        "rouge_l": mean([row["rouge_l"] for row in answerable]),
        "refusal_f1": mean([row["refusal_f1"] for row in irrelevant]),
        "recall_at_k": mean(recalls),
        "mrr": mean(ranks),
        "latency_ms": mean(latencies),
        "adapter_norm": mean(adapter_norms),
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
            gold_ids = set(qa.gold_atom_ids)
            ranked = result["ranked_atom_ids"]
            recall = rank_score = None
            if gold_ids:
                recall = len(gold_ids.intersection(ranked[:top_k])) / len(gold_ids)
                rank = next((i for i, atom_id in enumerate(ranked, 1) if atom_id in gold_ids), None)
                rank_score = 0.0 if rank is None else 1.0 / rank
            rows.append({
                "record_id": record.record_id,
                "split": record.split,
                "question_id": qa.question_id,
                "question": qa.question,
                "prediction": result["prediction"],
                "gold": list(qa.answers),
                "is_irrelevant": qa.is_irrelevant,
                "selected_atom_ids": result["selected_atom_ids"],
                "ranked_atom_ids": ranked,
                "routing_weights": result["routing_weights"],
                "recall_at_k": recall,
                "mrr": rank_score,
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
    args = parser.parse_args()

    if args.predictions:
        if any((args.config, args.data, args.checkpoint, args.output)):
            parser.error("--predictions cannot be combined with generation arguments")
        rows = [
            json.loads(line)
            for line in Path(args.predictions).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
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
        for row in rows:
            row["checkpoint_step"] = state.optimizer_step
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )

    summary = score_rows(rows)
    if not args.predictions:
        summary["checkpoint_step"] = state.optimizer_step
        summary["peak_vram_bytes"] = (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        )
        metrics_path = destination.parent / "metrics" / f"evaluation_{args.split}.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

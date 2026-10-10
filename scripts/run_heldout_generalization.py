from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from doc_to_atom.config import (
    load_config,
    load_project_env,
    require_disclosed_training_values,
    write_run_manifest,
)
from doc_to_atom.data.io import read_jsonl, validate_jsonl
from doc_to_atom.data.semantic_validation import audit_records
from doc_to_atom.evaluation.baselines import BaseQueryOnly, BaseWithContext
from doc_to_atom.evaluation.metrics import refusal_f1, rouge_l, word_f1
from doc_to_atom.inference.query import QueryEngine
from doc_to_atom.training.factory import build_system, parameter_counts
from doc_to_atom.training.loop import _memory, run_training


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _answer_text(answer: str | list[str] | None) -> str:
    if isinstance(answer, list):
        return str(answer[0]) if answer else ""
    return str(answer or "")


def _summarize(rows: list[dict], method: str, split: str) -> dict:
    subset = [row for row in rows if row["method"] == method and row["split"] == split]
    answerable = [row for row in subset if row["group"] == "answerable"]
    irrelevant = [row for row in subset if row["group"] == "generated_irrelevant"]
    summary = {
        "count": len(subset),
        "answer": {
            "count": len(answerable),
            "word_f1": _mean([row["word_f1"] for row in answerable]),
            "rouge_l": _mean([row["rouge_l"] for row in answerable]),
        },
        "irrelevant": {
            "count": len(irrelevant),
            "refusal_f1": _mean([row["refusal_f1"] for row in irrelevant]),
            "adapter_norm": _mean(
                [row["adapter_norm"] for row in irrelevant if row["adapter_norm"] is not None]
            ),
        },
        "relevant_adapter_norm": _mean(
            [row["adapter_norm"] for row in answerable if row["adapter_norm"] is not None]
        ),
        "query_latency_ms": _mean([row["query_latency_ms"] for row in subset]),
    }
    routed = [row for row in answerable if row["recall_at_1"] is not None]
    summary["routing"] = {
        "recall_at_1": _mean([row["recall_at_1"] for row in routed]),
        "recall_at_4": _mean([row["recall_at_4"] for row in routed]),
        "recall_at_8": _mean([row["recall_at_8"] for row in routed]),
        "mrr": _mean([row["mrr"] for row in routed]),
    }
    return summary


def _routing_metrics(ranked: list[str], gold: list[str]) -> tuple[float, float, float, float]:
    gold_set = set(gold)
    recall = lambda k: len(gold_set.intersection(ranked[:k])) / len(gold_set)
    first = next((rank for rank, atom_id in enumerate(ranked, 1) if atom_id in gold_set), None)
    return recall(1), recall(4), recall(8), 0.0 if first is None else 1.0 / first


@torch.inference_mode()
def _evaluate_base(system, tokenizer, records, method: str, max_new_tokens: int):
    baseline = (
        BaseQueryOnly(system.base_model, tokenizer, max_new_tokens=max_new_tokens)
        if method == "base_query_only"
        else BaseWithContext(system.base_model, tokenizer, max_new_tokens=max_new_tokens)
    )
    rows = []
    _sync()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for record in records:
        for qa in record.qa_pairs:
            _sync()
            started = time.perf_counter()
            prediction = baseline.answer(record.context, qa.question).strip()
            _sync()
            latency = (time.perf_counter() - started) * 1000
            target = "unanswerable" if qa.is_irrelevant else _answer_text(qa.answer)
            rows.append(
                {
                    "method": method,
                    "split": record.split,
                    "record_id": record.record_id,
                    "question_id": qa.question_id,
                    "question": qa.question,
                    "group": "generated_irrelevant" if qa.is_generated_irrelevant else "answerable",
                    "target": target,
                    "prediction": prediction,
                    "word_f1": word_f1(prediction, target) if not qa.is_irrelevant else None,
                    "rouge_l": rouge_l(prediction, target) if not qa.is_irrelevant else None,
                    "refusal_f1": refusal_f1(prediction) if qa.is_irrelevant else None,
                    "selected_atom_ids": [],
                    "routing_weights": [],
                    "recall_at_1": None,
                    "recall_at_4": None,
                    "recall_at_8": None,
                    "mrr": None,
                    "adapter_norm": None,
                    "query_latency_ms": latency,
                }
            )
    peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    return rows, {"compile_latency_ms": None, "peak_vram_bytes": peak}


@torch.inference_mode()
def _evaluate_doc2atom(system, tokenizer, records, config):
    engine = QueryEngine(
        system.base_model,
        tokenizer,
        system.encoder,
        system.router,
        system.composer,
        system.injector,
    )
    rows, compile_latencies = [], []
    _sync()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for record in records:
        _sync()
        started = time.perf_counter()
        bank = _memory(system, tokenizer, record, config)
        _sync()
        compile_latencies.append((time.perf_counter() - started) * 1000)
        for qa in record.qa_pairs:
            _sync()
            started = time.perf_counter()
            result = engine.answer(
                qa.question,
                bank,
                top_k=8,
                max_new_tokens=config["evaluation"]["max_new_tokens"],
            )
            _sync()
            latency = (time.perf_counter() - started) * 1000
            prediction = result["prediction"].strip()
            target = "unanswerable" if qa.is_irrelevant else _answer_text(qa.answer)
            route = (None, None, None, None)
            if not qa.is_irrelevant:
                route = _routing_metrics(result["ranked_atom_ids"], qa.gold_atom_ids)
            rows.append(
                {
                    "method": "doc2atom",
                    "split": record.split,
                    "record_id": record.record_id,
                    "question_id": qa.question_id,
                    "question": qa.question,
                    "group": "generated_irrelevant" if qa.is_generated_irrelevant else "answerable",
                    "target": target,
                    "prediction": prediction,
                    "word_f1": word_f1(prediction, target) if not qa.is_irrelevant else None,
                    "rouge_l": rouge_l(prediction, target) if not qa.is_irrelevant else None,
                    "refusal_f1": refusal_f1(prediction) if qa.is_irrelevant else None,
                    "selected_atom_ids": result["selected_atom_ids"],
                    "routing_weights": result["routing_weights"],
                    "ranked_atom_ids": result["ranked_atom_ids"],
                    "recall_at_1": route[0],
                    "recall_at_4": route[1],
                    "recall_at_8": route[2],
                    "mrr": route[3],
                    "adapter_norm": result["composed_adapter_norm"],
                    "query_latency_ms": latency,
                }
            )
    peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    return rows, {
        "compile_latency_ms": _mean(compile_latencies),
        "compile_document_count": len(compile_latencies),
        "peak_vram_bytes": peak,
    }


def main() -> None:
    load_project_env()
    parser = argparse.ArgumentParser(description="Small document-disjoint held-out experiment")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    require_disclosed_training_values(config)
    stats = validate_jsonl(args.data, reject_span_mismatch=True)
    records = read_jsonl(args.data)
    semantic = audit_records(records)
    if semantic["errors"]:
        raise ValueError(f"semantic validation failed: {semantic['errors']} errors")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "results.json").exists():
        raise FileExistsError(f"refusing to overwrite completed run: {output}")
    write_run_manifest(config, output / "run_manifest.json")

    system, tokenizer, targets = build_system(config)
    training = run_training(system, tokenizer, records, config, output)
    system.eval()

    # Test is not passed to the trainer's validation path and is evaluated only here,
    # after optimization has ended. No metric below changes configuration or weights.
    ordered = [
        record
        for split in ("train", "validation", "test")
        for record in records
        if record.split == split
    ]
    rows = []
    efficiency = {}
    for method in ("base_query_only", "base_context"):
        method_rows, method_efficiency = _evaluate_base(
            system, tokenizer, ordered, method, config["evaluation"]["max_new_tokens"]
        )
        rows.extend(method_rows)
        efficiency[method] = method_efficiency
    method_rows, method_efficiency = _evaluate_doc2atom(system, tokenizer, ordered, config)
    rows.extend(method_rows)
    efficiency["doc2atom"] = method_efficiency

    summaries = {
        method: {
            split: _summarize(rows, method, split) for split in ("train", "validation", "test")
        }
        for method in ("base_query_only", "base_context", "doc2atom")
    }
    result = {
        "data": stats,
        "semantic_validation": semantic,
        "parameters": parameter_counts(system),
        "target_count": len(targets),
        "device": str(next(system.parameters()).device),
        "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "training": training,
        "summaries": summaries,
        "efficiency": efficiency,
        "test_policy": "evaluated once after training; never used for tuning or early stopping",
    }
    (output / "predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    (output / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

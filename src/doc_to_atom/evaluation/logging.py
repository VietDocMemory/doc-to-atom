from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path

import torch


@contextmanager
def measured_cuda():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = {}
    try:
        yield result
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            result["cuda_peak_memory_mb"] = torch.cuda.max_memory_allocated() / 1024**2
        else:
            result["cuda_peak_memory_mb"] = 0.0
        result["latency_seconds"] = time.perf_counter() - start


def append_example(path: str | Path, row: dict) -> None:
    required = {
        "selected_atom_ids",
        "stage1_scores",
        "reranker_scores",
        "metadata_biases",
        "routing_weights",
        "composed_adapter_norm",
        "prediction",
        "gold",
        "f1",
        "rouge_l",
        "latency_seconds",
        "cuda_peak_memory_mb",
    }
    missing = required - row.keys()
    if missing:
        raise ValueError(f"missing per-example fields: {sorted(missing)}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


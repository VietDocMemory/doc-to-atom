"""Named benchmark sources used by the D2A paper-style workflow."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkDataset:
    """A public source and the parser needed to turn a row into QA input."""

    name: str
    hf_dataset: str
    hf_config: str | None
    hf_split: str
    preprocessor: str


def _longbench(name: str) -> BenchmarkDataset:
    return BenchmarkDataset(f"longbench_{name}", "THUDM/LongBench", name, "test", "longbench")


# The three short-context entries match the public source identifiers used by
# doc-to-lora. LongBench task names are official THUDM/LongBench configs.
# Standalone 2WikiMultihopQA/QASPER are intentionally not auto-downloaded:
# their source releases are not pinned by the paper, so use a pinned JSONL
# export with prepare_benchmark.py rather than silently choosing a version.
BENCHMARK_CATALOG: dict[str, BenchmarkDataset] = {
    "squad": BenchmarkDataset("squad", "rajpurkar/squad", None, "validation", "squad"),
    "drop": BenchmarkDataset("drop", "ucinlp/drop", None, "validation", "drop"),
    "ropes": BenchmarkDataset("ropes", "allenai/ropes", None, "validation", "ropes"),
    **{
        f"longbench_{name}": _longbench(name)
        for name in (
            "2wikimqa", "hotpotqa", "multifieldqa_en", "multifieldqa_zh",
            "musique", "narrativeqa", "qasper", "triviaqa",
        )
    },
}


def get_benchmark_dataset(name: str) -> BenchmarkDataset:
    """Resolve a D2A benchmark alias or explain how to provide custom data."""
    try:
        return BENCHMARK_CATALOG[name]
    except KeyError as exc:
        supported = ", ".join(sorted(BENCHMARK_CATALOG))
        raise ValueError(
            f"unsupported automatic benchmark dataset '{name}'. Supported: {supported}. "
            "For a pinned custom release, use prepare_benchmark.py with --source-jsonl."
        ) from exc

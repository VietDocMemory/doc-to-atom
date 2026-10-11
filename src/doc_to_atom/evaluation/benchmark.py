"""Reusable scoring and reporting utilities for paper-style QA benchmarks.

The benchmark runner deliberately keeps model execution separate from scoring: this
makes a Doc2Atom run and externally-produced Doc-to-LoRA predictions comparable
without pretending that this repository implements Doc-to-LoRA.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from .metrics import rouge_l, word_f1


PAPER_METHOD_ORDER = ("base_context", "d2l_pt", "d2l_raw", "doc2atom")
PAPER_METHOD_LABELS = {
    "base_context": "Base+ctx",
    "d2l_pt": "D2L-pt",
    "d2l_raw": "D2L-raw",
    "doc2atom": "Doc2Atom",
}


def score_row(row: dict[str, Any]) -> dict[str, Any]:
    """Return a prediction row with normalized metrics on the 0--1 scale."""
    result = dict(row)
    gold = result.get("gold", result.get("target", ""))
    if isinstance(gold, list):
        # SQuAD-like annotations may contain several equally-valid references.
        references = [str(item) for item in gold] or [""]
    else:
        references = [str(gold or "")]
    prediction = str(result.get("prediction", ""))
    result["f1"] = max(word_f1(prediction, reference) for reference in references)
    result["rouge_l"] = max(rouge_l(prediction, reference) for reference in references)
    return result


def score_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [score_row(row) for row in rows]


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate metrics by model, method, dataset and split.

    Scores are retained as fractions in JSON and converted to percentages only in
    the rendered table, avoiding accidental double scaling.
    """
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        # The paper's F1/ROUGE-L tables are answer-quality metrics.  Generated
        # irrelevant probes are retained in scored_predictions.jsonl for refusal
        # analysis but must not be mixed into these answerable QA averages.
        if row.get("is_irrelevant") or row.get("group") == "generated_irrelevant":
            continue
        required = ("model", "method", "dataset", "split")
        missing = [field for field in required if not row.get(field)]
        if missing:
            raise ValueError(f"prediction row missing benchmark fields: {', '.join(missing)}")
        groups[tuple(str(row[field]) for field in required)].append(row)

    summaries = []
    for (model, method, dataset, split), group in sorted(groups.items()):
        summaries.append(
            {
                "model": model,
                "method": method,
                "dataset": dataset,
                "split": split,
                "count": len(group),
                "f1": statistics.fmean(float(row["f1"]) for row in group),
                "rouge_l": statistics.fmean(float(row["rouge_l"]) for row in group),
            }
        )
    return summaries


def render_markdown_table(summaries: list[dict[str, Any]], metric: str, split: str = "test") -> str:
    """Render the paper layout, grouping method columns below each base model."""
    filtered = [row for row in summaries if row["split"] == split]
    if not filtered:
        return f"No {split} rows available."
    models = sorted({str(row["model"]) for row in filtered})
    datasets = sorted({str(row["dataset"]) for row in filtered})
    lookup = {(r["dataset"], r["model"], r["method"]): r for r in filtered}
    header = ["Dataset"] + [f"{model} {PAPER_METHOD_LABELS[method]}" for model in models for method in PAPER_METHOD_ORDER]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for dataset in datasets:
        values = [dataset]
        for model in models:
            for method in PAPER_METHOD_ORDER:
                row = lookup.get((dataset, model, method))
                values.append("—" if row is None else f"{100 * float(row[metric]):.2f}")
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def read_prediction_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_report(rows: list[dict[str, Any]], output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    scored = score_rows(rows)
    summaries = summarize_rows(scored)
    (output / "scored_predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in scored), encoding="utf-8"
    )
    result = {
        "summary": summaries,
        "f1_table": render_markdown_table(summaries, "f1"),
        "rouge_l_table": render_markdown_table(summaries, "rouge_l"),
    }
    (output / "benchmark_results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output / "benchmark_tables.md").write_text(
        "# F1\n\n" + result["f1_table"] + "\n\n# ROUGE-L\n\n" + result["rouge_l_table"] + "\n",
        encoding="utf-8",
    )
    return result

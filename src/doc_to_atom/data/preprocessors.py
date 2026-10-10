from __future__ import annotations

from collections.abc import Callable
from typing import Any


def _answers(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return list(value.get("text", []))
    return [str(item) for item in value]


def squad(row: dict[str, Any]) -> dict[str, Any]:
    return {"context": row["context"], "question": row["question"], "answers": _answers(row["answers"])}


def drop(row: dict[str, Any]) -> dict[str, Any]:
    answer = row.get("answers_spans") or row.get("answer")
    return {"context": row["passage"], "question": row["question"], "answers": _answers(answer)}


def ropes(row: dict[str, Any]) -> dict[str, Any]:
    context = f"{row.get('background', '')}\n{row.get('situation', '')}".strip()
    return {"context": context, "question": row["question"], "answers": _answers(row.get("answers"))}


def wiki2(row: dict[str, Any]) -> dict[str, Any]:
    context = row.get("context", "")
    if isinstance(context, list):
        passages = []
        for item in context:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                title, sentences = item
                passages.append(f"Passage [{title}]: {' '.join(sentences)}")
        context = "\n".join(passages)
    return {"context": context, "question": row["question"], "answers": _answers(row.get("answer"))}


def qasper(row: dict[str, Any]) -> dict[str, Any]:
    full_text = row.get("full_text", row.get("context", ""))
    if isinstance(full_text, dict):
        paragraphs = full_text.get("paragraphs", [])
        context = "\n".join(" ".join(section) if isinstance(section, list) else str(section) for section in paragraphs)
    else:
        context = str(full_text)
    return {"context": context, "question": row.get("question", ""), "answers": _answers(row.get("answers"))}


def longbench(row: dict[str, Any]) -> dict[str, Any]:
    return {"context": row["context"], "question": row["input"], "answers": _answers(row.get("answers"))}


DATASET_PREPROCESSORS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "squad": squad,
    "drop": drop,
    "ropes": ropes,
    "2wikimultihopqa": wiki2,
    "qasper": qasper,
    "longbench": longbench,
}


def preprocess_dataset_row(dataset: str, row: dict[str, Any]) -> dict[str, Any]:
    try:
        return DATASET_PREPROCESSORS[dataset.lower()](row)
    except KeyError as exc:
        raise ValueError(f"unsupported paper dataset {dataset}") from exc


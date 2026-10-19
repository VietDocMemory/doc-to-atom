from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from doc_to_atom.config import load_project_env

from .client import MiniMaxAtomizer, MiniMaxConfig, OpenAIAtomizer, OpenAIConfig
from .models import QuestionAnnotation, StructuredRecord
from .provenance import write_raw_response_jsonl

LOGGER = logging.getLogger(__name__)


def _source_answers(item: dict) -> list[str]:
    value = item["answers"] if "answers" in item else item.get("answer", [])
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list) or any(not isinstance(answer, str) for answer in value):
        raise ValueError("source answers must be a string or an ordered list of strings")
    return list(value)


def _source_question_id(item: dict) -> str:
    value = item.get("id", item.get("question_id"))
    if value is None or not str(value).strip():
        raise ValueError("source question is missing question_id")
    return str(value)


def main() -> None:
    load_project_env()
    parser = argparse.ArgumentParser(description="Offline three-stage XML semantic atomization")
    parser.add_argument("--input", required=True, help="JSONL with id, context, questions")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--raw-responses",
        help="Append-only raw provider response JSONL (default: <output>.raw-responses.jsonl)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append missing records and reuse existing cached record IDs",
    )
    parser.add_argument(
        "--retry-invalid",
        action="store_true",
        help="With --resume, discard cached records containing invalid spans and re-atomize them",
    )
    parser.add_argument(
        "--provider",
        choices=["minimax", "openai"],
        default=os.getenv("ANNOTATION_PROVIDER", "minimax"),
    )
    args = parser.parse_args()
    atomizer = (
        OpenAIAtomizer(OpenAIConfig.from_env())
        if args.provider == "openai"
        else MiniMaxAtomizer(MiniMaxConfig.from_env())
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_responses = Path(args.raw_responses or f"{output}.raw-responses.jsonl")
    if not args.resume:
        raw_responses.parent.mkdir(parents=True, exist_ok=True)
        raw_responses.open("w", encoding="utf-8").close()
    atomizer.response_sink = lambda response: write_raw_response_jsonl(response, raw_responses)
    cached_ids: set[str] = set()
    retained_lines: list[str] = []
    if args.resume and output.exists():
        with output.open(encoding="utf-8") as cached:
            for line in cached:
                if not line.strip():
                    continue
                record = StructuredRecord.model_validate_json(line)
                if args.retry_invalid and any(not atom.span_valid for atom in record.atoms):
                    continue
                cached_ids.add(record.record_id)
                retained_lines.append(line.rstrip("\n"))
        if args.retry_invalid:
            with output.open("w", encoding="utf-8", newline="\n") as rewritten:
                for line in retained_lines:
                    rewritten.write(line + "\n")
    failed = 0
    mode = "a" if args.resume else "w"
    with (
        Path(args.input).open(encoding="utf-8") as source,
        output.open(mode, encoding="utf-8") as sink,
    ):
        for index, line in enumerate(source):
            raw = json.loads(line)
            record_id = str(raw.get("id", index))
            if record_id in cached_ids:
                continue
            questions = [
                QuestionAnnotation(
                    question_id=_source_question_id(item),
                    question=item["question"],
                    answers=_source_answers(item),
                    answer_provenance="source",
                    question_type=item.get("question_type", "single_hop"),
                    is_irrelevant=bool(item.get("is_irrelevant", False)),
                )
                for item in raw.get("questions", [])
            ]
            record = atomizer.atomize(
                record_id,
                raw["context"],
                questions,
                raw.get("dataset"),
                raw.get("split"),
            )
            if record is not None:
                sink.write(record.model_dump_json() + "\n")
            else:
                failed += 1
                LOGGER.error(
                    "record %s discarded during atomization: %s",
                    raw.get("id", index),
                    "; ".join(atomizer.last_errors) or "unknown error",
                )
    if failed:
        raise RuntimeError(f"atomization discarded {failed} record(s); output is incomplete")

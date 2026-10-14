from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from doc_to_atom.config import load_project_env

from .client import (
    GeminiAtomizer,
    GeminiConfig,
    GroqAtomizer,
    GroqConfig,
    MiniMaxAtomizer,
    MiniMaxConfig,
    OpenAIAtomizer,
    OpenAIConfig,
)
from .models import QuestionAnnotation, StructuredRecord

LOGGER = logging.getLogger(__name__)


def main() -> None:
    load_project_env()
    parser = argparse.ArgumentParser(description="Offline three-stage XML semantic atomization")
    parser.add_argument("--input", required=True, help="JSONL with id, context, questions")
    parser.add_argument("--output", required=True)
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
        choices=["minimax", "openai", "gemini", "groq"],
        default=os.getenv("ANNOTATION_PROVIDER", "minimax"),
    )
    args = parser.parse_args()
    atomizer = (
        OpenAIAtomizer(OpenAIConfig.from_env())
        if args.provider == "openai"
        else GeminiAtomizer(GeminiConfig.from_env())
        if args.provider == "gemini"
        else GroqAtomizer(GroqConfig.from_env())
        if args.provider == "groq"
        else MiniMaxAtomizer(MiniMaxConfig.from_env())
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
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
                    question_id=str(item.get("id", f"q_{q_index}")),
                    question=item["question"],
                    answer=item.get("answer"),
                    question_type=item.get("question_type", "single_hop"),
                )
                for q_index, item in enumerate(raw.get("questions", []))
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

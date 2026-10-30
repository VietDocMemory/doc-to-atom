from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import tempfile
from dataclasses import replace
from pathlib import Path

from doc_to_atom.config import load_project_env

from .client import DEFAULT_MINIMAX_MODEL, DEFAULT_REASONING_EFFORT, MiniMaxAtomizer, MiniMaxConfig
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
        help="With --resume, snapshot the old output, replace invalid-span records and re-atomize them",
    )
    parser.add_argument(
        "--provider",
        choices=["minimax"],
        default="minimax",
        help="Only MiniMax is supported (flag retained for existing MiniMax commands)",
    )
    parser.add_argument("--model", default=DEFAULT_MINIMAX_MODEL)
    parser.add_argument("--max-completion-tokens", type=int, default=8192)
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high", "xhigh", "max"], default=DEFAULT_REASONING_EFFORT)
    args = parser.parse_args()
    print(json.dumps(atomize_file(
        args.input, args.output, provider=args.provider, resume=args.resume,
        retry_invalid=args.retry_invalid, raw_responses=args.raw_responses,
        model=args.model, max_completion_tokens=args.max_completion_tokens,
        reasoning_effort=args.reasoning_effort,
    ), indent=2))


def atomize_file(
    input_path: str | Path, output_path: str | Path, *, provider: str = "minimax",
    resume: bool = True, retry_invalid: bool = False,
    raw_responses: str | Path | None = None,
    model: str = DEFAULT_MINIMAX_MODEL, max_completion_tokens: int = 8192,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
) -> dict:
    """Atomize source JSONL in-process; resume appends only uncached record IDs."""
    if provider != "minimax":
        raise ValueError(f"unsupported annotation provider: {provider}")
    atomizer = None  # Cache-only resume does not require credentials or a network request.
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_path = Path(raw_responses or f"{output}.raw-responses.jsonl")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if not resume:
        if output.exists() or raw_path.exists():
            raise FileExistsError("refusing to overwrite existing atomization output or raw responses")
        raw_path.touch()
    cached_ids: set[str] = set()
    retained_lines: list[str] = []
    removed_invalid = False
    if resume and output.exists():
        with output.open(encoding="utf-8") as cached:
            for line in cached:
                if not line.strip():
                    continue
                record = StructuredRecord.model_validate_json(line)
                if retry_invalid and any(not atom.span_valid for atom in record.atoms):
                    removed_invalid = True
                    continue
                cached_ids.add(record.record_id)
                retained_lines.append(line.rstrip("\n"))
        if retry_invalid and removed_invalid:
            original = output.read_bytes()
            replacement = ("\n".join(retained_lines) + ("\n" if retained_lines else "")).encode("utf-8")
            if replacement != original:
                # Explicit retries may replace invalid records, but never destroy
                # the previous research artifact or its append-only raw sidecar.
                digest = hashlib.sha256(original).hexdigest()
                backup = output.with_name(f"{output.name}.before-retry-{digest}.jsonl")
                if backup.exists():
                    if backup.read_bytes() != original:
                        raise ValueError(f"retry backup differs: {backup}")
                else:
                    with backup.open("xb") as snapshot:
                        snapshot.write(original)
                with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as rewritten:
                    temporary = Path(rewritten.name)
                    rewritten.write(replacement)
                try:
                    os.replace(temporary, output)
                finally:
                    temporary.unlink(missing_ok=True)
        LOGGER.warning("Cached records are reused unchanged, including historical provider/model provenance")
    failed = 0
    mode = "a" if resume else "x"
    with (
        Path(input_path).open(encoding="utf-8") as source,
        output.open(mode, encoding="utf-8") as sink,
    ):
        for index, line in enumerate(source):
            raw = json.loads(line)
            record_id = str(raw.get("id", index))
            if record_id in cached_ids:
                continue
            if atomizer is None:
                config = replace(MiniMaxConfig.from_env(), model=model,
                                 max_completion_tokens=max_completion_tokens,
                                 reasoning_effort=reasoning_effort)
                atomizer = MiniMaxAtomizer(config)
                atomizer.response_sink = lambda response: write_raw_response_jsonl(response, raw_path)
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
    return {"records_cached": len(cached_ids), "records_failed": failed,
            "output": str(output), "provider": provider}

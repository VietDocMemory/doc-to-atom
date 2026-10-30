"""Paired effort pilot, source-aware release gate, and provenance-preserving production run.

Run with uv run scripts/run_annotation_experiment.py --source ... --output-dir ...
An existing output directory is refused: resubmission must never silently repeat paid calls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from doc_to_atom.atomization.cli import _source_answers, _source_question_id
from doc_to_atom.atomization.client import MiniMaxAtomizer, MiniMaxConfig
from doc_to_atom.atomization.models import QuestionAnnotation
from doc_to_atom.atomization.parser import parse_question_roles_xml
from doc_to_atom.atomization.prompts import (
    DATASET_ADAPTERS, DECOMPOSITION_SYSTEM, IRRELEVANT_SYSTEM, QUESTION_SYSTEM,
)
from doc_to_atom.atomization.provenance import RawProviderResponse, write_raw_response_jsonl
from doc_to_atom.atomization.validation import validate_record
from doc_to_atom.config import load_project_env
from doc_to_atom.data.io import read_jsonl
from doc_to_atom.data.semantic_validation import audit_records, load_source_records

EFFORTS = ("low", "medium", "high", "xhigh", "max")


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def raw_metrics(path: Path) -> dict:
    """Measure provider output BEFORE canonicalization; clean final labels alone can mislead."""
    stages: Counter = Counter()
    prompt = completion = thinking = calls = role_questions = overlaps = parse_notices = 0
    latency = 0.0
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = RawProviderResponse.model_validate_json(line)
        body = json.loads(raw.raw_body or "{}")
        usage = body.get("usage") or {}
        calls += 1
        prompt += usage.get("prompt_tokens", 0)
        completion += usage.get("completion_tokens", 0)
        thinking += (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
        latency += raw.latency_ms or 0
        stages[raw.stage] += 1
        if raw.stage == "question_annotation":
            roles, notices = parse_question_roles_xml(raw.content)
            parse_notices += len(notices)
            for role in roles:
                role_questions += 1
                gold, support, distract = map(set, (
                    role.gold_atom_ids, role.supporting_atom_ids, role.distractor_atom_ids,
                ))
                overlaps += bool(gold & support or gold & distract or support & distract)
    return {
        "api_calls": calls, "stage_calls": dict(stages), "prompt_tokens": prompt,
        "completion_tokens": completion, "reasoning_tokens": thinking,
        "total_tokens": prompt + completion, "summed_request_latency_ms": latency,
        "provider_role_questions": role_questions, "provider_overlap_questions": overlaps,
        "provider_overlap_rate": overlaps / role_questions if role_questions else None,
        "provider_role_parse_notices": parse_notices,
    }


def source_report(directory: Path) -> dict:
    records = read_jsonl(directory / "atoms.jsonl")
    for record in records:
        validate_record(record)
    sources = load_source_records(directory / "raw_source.jsonl")
    report = audit_records(records, sources)
    report["atoms"] = sum(len(r.atoms) for r in records)
    report["invalid_spans"] = sum(not a.span_valid for r in records for a in r.atoms)
    report["context_mismatches"] = sum(r.context != sources[r.record_id]["context"] for r in records)
    report["split_mismatches"] = sum(r.split != sources[r.record_id].get("split") for r in records)
    report["accepted"] = (
        bool(records) and not report["errors"] and not report["invalid_spans"]
        and not report["context_mismatches"] and not report["split_mismatches"]
        and report["answer_preservation_rate"] == 1.0
        and not report["provider_unknown_atom_references"]
    )
    save_json(directory / "validation.json", report)
    return report


def annotate(directory: Path, lines: list[str], config: MiniMaxConfig,
             cached: Path | None = None) -> dict:
    directory.mkdir(parents=True)
    (directory / "raw_source.jsonl").write_text("".join(lines), encoding="utf-8")
    atoms_path = directory / "atoms.jsonl"
    raw_path = directory / "raw_responses.jsonl"
    if cached:
        shutil.copyfile(cached / "atoms.jsonl", atoms_path)
        shutil.copyfile(cached / "raw_responses.jsonl", raw_path)
    else:
        atoms_path.touch()
        raw_path.touch()
    cached_records = {r.record_id for r in read_jsonl(atoms_path)}
    atomizer = MiniMaxAtomizer(config)
    atomizer.response_sink = lambda r: write_raw_response_jsonl(r, raw_path)
    started = time.perf_counter()
    failures = []
    for line in lines:
        row = json.loads(line)
        if str(row["id"]) in cached_records:
            continue
        questions = [QuestionAnnotation(
            question_id=_source_question_id(q), question=q["question"],
            answers=_source_answers(q), answer_provenance="source",
            question_type=q.get("question_type", "single_hop"),
            is_irrelevant=bool(q.get("is_irrelevant", False)),
        ) for q in row.get("questions", [])]
        print(f"{directory.name}: {row['id']} starting", flush=True)
        try:
            record = atomizer.atomize(str(row["id"]), row["context"], questions,
                                     row.get("dataset"), row.get("split"))
            if record is None:
                failures.append({"record_id": row["id"], "errors": atomizer.last_errors})
            else:
                with atoms_path.open("a", encoding="utf-8") as sink:
                    sink.write(record.model_dump_json() + "\n")
        except Exception as exc:
            # Do not persist exception messages: SDK/provider diagnostics can contain secrets.
            failures.append({"record_id": row["id"], "error_type": type(exc).__name__})
            print(f"{directory.name}: stopped after {type(exc).__name__}; no retry", flush=True)
            break
        print(f"{directory.name}: {row['id']} finished", flush=True)
    report = source_report(directory)
    metrics = {
        "reasoning_effort": config.reasoning_effort,
        "wall_time_seconds_new_work": time.perf_counter() - started,
        "cached_records": len(cached_records), "failures": failures,
        **raw_metrics(raw_path), "validation": report,
    }
    save_json(directory / "metrics.json", metrics)
    return metrics


def choose_effort(results: list[dict]) -> dict:
    eligible = [r for r in results if r["validation"]["accepted"] and not r["failures"]]
    if not eligible:
        raise ValueError("No effort passed source-aware gates; production is blocked")
    # Mechanical validity first, then raw role cleanliness, then measured cost/latency.
    return min(eligible, key=lambda r: (
        r["provider_overlap_questions"], r["provider_role_parse_notices"],
        r["total_tokens"], r["summed_request_latency_ms"],
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--pilot-documents", type=int, default=2)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    load_project_env()
    config = replace(MiniMaxConfig.from_env(), max_completion_tokens=524288,
                     timeout_seconds=1800)
    lines = [s.rstrip("\r\n") + "\n" for s in args.source.read_text(encoding="utf-8").splitlines()
             if s.strip()]
    sources = load_source_records(args.source)  # Reject duplicate IDs before spending.
    if any(row.get("split") not in {"train", "validation"} for row in sources.values()):
        raise ValueError("Source must have explicit train/validation splits")
    train_lines = [s for s in lines if json.loads(s)["split"] == "train"]
    if not 1 <= args.pilot_documents <= len(train_lines):
        raise ValueError("Pilot size must fit the train split")
    # Do not tune effort on validation documents.
    pilot = train_lines[:args.pilot_documents]
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.source, root / "raw_source.jsonl")
    prompts = {"decomposition": DECOMPOSITION_SYSTEM, "irrelevant": IRRELEVANT_SYSTEM,
               "question": QUESTION_SYSTEM, "dataset_adapters": DATASET_ADAPTERS}
    save_json(root / "prompts.json", prompts)
    save_json(root / "manifest.json", {
        "source": str(args.source), "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "model": config.model, "max_completion_tokens": config.max_completion_tokens,
        "timeout_seconds": config.timeout_seconds, "efforts": EFFORTS,
        "pilot_record_ids": [json.loads(s)["id"] for s in pilot], "pilot_split": "train",
        "prompt_sha256": hashlib.sha256(json.dumps(prompts, sort_keys=True).encode()).hexdigest(),
        "split_counts": dict(Counter(row["split"] for row in sources.values())),
        "limitations": "One trial per effort; automatic consistency/grounding proxies, not human accuracy. "
                        "Warm caches and concurrent requests can affect latency and billed input cost.",
    })
    def trial(effort):
        return annotate(root / "benchmark" / effort, pilot,
                        replace(config, reasoning_effort=effort))

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(trial, EFFORTS))
    save_json(root / "comparison.json", {"results": results})
    selected = choose_effort(results)
    effort = selected["reasoning_effort"]
    save_json(root / "selection.json", {
        "reasoning_effort": effort,
        "rule": "source-aware gates, then raw overlaps, parse notices, total tokens, latency",
        "validation_used_for_selection": False,
    })
    production = root / "production"
    annotate(production, lines, replace(config, reasoning_effort=effort),
             cached=root / "benchmark" / effort)
    # Preserve paired artifacts per split; downstream training must use an accepted release.
    records = read_jsonl(production / "atoms.jsonl")
    raw = [RawProviderResponse.model_validate_json(s) for s in
           (production / "raw_responses.jsonl").read_text(encoding="utf-8").splitlines()]
    split_reports = {}
    for split in ("train", "validation"):
        folder = production / split
        folder.mkdir()
        subset = [s for s in lines if json.loads(s)["split"] == split]
        ids = {str(json.loads(s)["id"]) for s in subset}
        (folder / "raw_source.jsonl").write_text("".join(subset), encoding="utf-8")
        (folder / "atoms.jsonl").write_text("".join(
            r.model_dump_json() + "\n" for r in records if r.record_id in ids), encoding="utf-8")
        (folder / "raw_responses.jsonl").write_text("".join(
            r.model_dump_json() + "\n" for r in raw if r.record_id in ids), encoding="utf-8")
        split_reports[split] = source_report(folder)
    accepted = all(r["accepted"] for r in split_reports.values())
    save_json(root / "release.json", {
        "training_ready": accepted, "reasoning_effort": effort,
        "production": str(production), "splits": split_reports,
    })
    print(json.dumps({"training_ready": accepted, "effort": effort,
                      "production": str(production)}, indent=2), flush=True)
    if not accepted:
        raise SystemExit("Source-aware gates failed; do not train on this release")


if __name__ == "__main__":
    main()

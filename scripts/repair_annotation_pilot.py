"""Repair the saved validation record; never regenerate valid training records.

Requires --retry-question-annotation to authorize exactly one paid stage retry.
All original pilot files remain untouched. Existing repair destinations are refused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from doc_to_atom.atomization.client import MiniMaxAtomizer, MiniMaxConfig
from doc_to_atom.atomization.models import QuestionAnnotation, TransformationEvent
from doc_to_atom.atomization.parser import parse_atoms_xml, parse_question_roles_xml
from doc_to_atom.atomization.prompts import QUESTION_SYSTEM
from doc_to_atom.atomization.provenance import RawProviderResponse, write_raw_response_jsonl
from doc_to_atom.atomization.question_join import build_authoritative_questions, join_question_roles
from doc_to_atom.atomization.validation import validate_record
from doc_to_atom.config import load_project_env
from doc_to_atom.data.io import read_jsonl
from doc_to_atom.data.semantic_validation import audit_records, load_source_records

TARGET = "squad_validation_0"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def snapshot(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): digest(p.read_bytes())
            for p in sorted(root.rglob("*")) if p.is_file()}


def strict_report(records, sources) -> dict:
    """Audit without canonicalization: never fix an invalid label just to pass a gate."""
    report = audit_records(records, sources)
    extra = []
    for r in records:
        source = sources.get(r.record_id)
        if source:
            for field in ("context", "split", "dataset"):
                if getattr(r, field) != source.get(field):
                    extra.append({"record_id": r.record_id, "code": f"source_{field}_mismatch"})
        ids = [a.atom_id for a in r.atoms]
        if len(ids) != len(set(ids)) or not ids:
            extra.append({"record_id": r.record_id, "code": "invalid_atom_bank"})
        for a in r.atoms:
            if not a.source_span or a.source_span not in r.context or not a.span_valid:
                extra.append({"record_id": r.record_id, "code": "strict_span_mismatch", "atom_id": a.atom_id})
            for rel in a.relations:
                if rel.target_atom_id not in ids:
                    extra.append({"record_id": r.record_id, "code": "unknown_relation_target", "atom_id": a.atom_id})
        for q in r.qa_pairs:
            for role in (q.gold_atom_ids, q.supporting_atom_ids, q.distractor_atom_ids):
                if len(role) != len(set(role)):
                    extra.append({"record_id": r.record_id, "code": "duplicate_role_id", "question_id": q.question_id})
        expected_counts = (len(r.atoms), len(r.qa_pairs), sum(q.is_generated_irrelevant for q in r.qa_pairs))
        actual_counts = (r.annotation_meta.num_atoms, r.annotation_meta.num_questions,
                         r.annotation_meta.num_irrelevant_generated)
        if expected_counts != actual_counts:
            extra.append({"record_id": r.record_id, "code": "metadata_count_mismatch"})
        if r.annotation_meta.errors:
            extra.append({"record_id": r.record_id, "code": "annotation_errors"})
    report["issues"].extend({**issue, "severity": "error"} for issue in extra)
    report["errors"] += len(extra)
    report["atoms"] = sum(len(r.atoms) for r in records)
    report["invalid_spans"] = sum(a.source_span not in r.context or not a.span_valid
                                  for r in records for a in r.atoms)
    report["accepted"] = (bool(records) and not report["errors"]
                          and not report["provider_unknown_atom_references"]
                          and report["answer_preservation_rate"] == 1.0)
    return report


def recover_probe_xml(text: str) -> tuple[str, list[str]]:
    # Narrow recovery only: one missing root '<', complete closing root, no text loss.
    if not text.startswith("irrelevant_questions>"):
        raise ValueError("Probe response does not match the reviewed one-character defect")
    repaired = "<" + text
    root = ET.fromstring(repaired)
    if root.tag != "irrelevant_questions" or len(root) != 2:
        raise ValueError("Unexpected probe structure")
    if any(q.tag != "question" or list(q) or not (q.text or "").strip() for q in root):
        raise ValueError("Unexpected probe payload")
    questions = [q.text.strip() for q in root]
    expected = ["Which player was named Most Valuable Player of Super Bowl 50?",
                "Who performed the halftime show at Super Bowl 50?"]
    if questions != expected:
        raise ValueError("Probes differ from source-reviewed pilot questions")
    return repaired, questions


def prepare_record(original, source, responses):
    record = original.model_copy(deep=True)
    events = []

    def event(code, before, after, **details):
        events.append({"code": code, "stage": "pilot_repair", "before": before,
                       "after": after, **details})

    probe_text, probe_questions = recover_probe_xml(responses["irrelevant_probe"].content)
    event("INSERT_MISSING_XML_ROOT_DELIMITER", responses["irrelevant_probe"].content,
          probe_text, reason="One leading character only; strict XML parse succeeds without losing text")
    decomp = responses["decomposition"].content
    if decomp.count("</emantic_type>") != 1:
        raise ValueError("Unexpected decomposition XML; refusing broad recovery")
    corrected = decomp.replace("</emantic_type>", "</semantic_type>")
    event("REPAIR_SEMANTIC_TYPE_CLOSING_TAG", "</emantic_type>", "</semantic_type>", atom_id="atom_12")
    root = ET.fromstring(corrected)
    for node in root.findall("atom"):
        relations = node.find("relations")
        if relations is not None:
            for edge in list(relations):
                if edge.get("type") == "participates":
                    event("DROP_UNSUPPORTED_RELATION", ET.tostring(edge, encoding="unicode"), None,
                          atom_id=node.findtext("atom_id"),
                          reason="Unsupported edge is removed, not mapped to a different semantic relation")
                    relations.remove(edge)
    derived_decomp = ET.tostring(root, encoding="unicode")
    atoms, warnings = parse_atoms_xml(derived_decomp)
    if warnings or len(atoms) != 12 or any(a.source_span not in source["context"] for a in atoms):
        raise ValueError("Recovered atom bank failed exact source/schema checks")
    for atom in atoms:
        if atom.atom_id == "atom_8":
            before = atom.source_span
            start = source["context"].index(before)
            after = source["context"][start:start + len(before) + 1]
            if after != before + ".":
                raise ValueError("Expected source-backed terminal punctuation is absent")
            atom.source_span = after
            event("EXTEND_SOURCE_SPAN_TERMINAL_PUNCTUATION", before, after, atom_id="atom_8",
                  source_start=start, source_end=start + len(after),
                  reason="Preserve all original answer variants, including terminal period")
    event("RESTORE_SAVED_ATOM_BANK", [a.model_dump() for a in record.atoms],
          [a.model_dump() for a in atoms], restored_ids=["atom_5", "atom_6", "atom_12"])
    record.atoms = atoms
    probes = [QuestionAnnotation(question_id=f"generated_irrelevant_{i}", question=text,
                                 question_type="irrelevant", answer_provenance="generated_irrelevant",
                                 is_irrelevant=True, is_generated_irrelevant=True)
              for i, text in enumerate(probe_questions)]
    return record, probes, events, probe_text, derived_decomp


def usage(path: Path) -> dict:
    tokens = Counter()
    latency = 0.0
    calls = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = RawProviderResponse.model_validate_json(line)
        body = json.loads(raw.raw_body or "{}")
        u = body.get("usage") or {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            tokens[key] += u.get(key, 0)
        tokens["reasoning_tokens"] += (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
        tokens["cached_prompt_tokens"] += (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        calls += 1
        latency += raw.latency_ms or 0
    return {"api_calls": calls, **tokens, "summed_request_latency_ms": latency}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--retry-question-annotation", action="store_true", required=True)
    args = parser.parse_args()
    pilot, out = args.pilot.resolve(), args.output_dir.resolve()
    if out == pilot or pilot in out.parents:
        raise ValueError("Repair destination must be outside original pilot")
    original_hashes = snapshot(pilot)
    production = pilot / "production"
    sources = load_source_records(production / "raw_source.jsonl")
    authoritative = load_source_records(pilot / "raw_source.jsonl")
    parent_manifest = json.loads((pilot / "manifest.json").read_text(encoding="utf-8"))
    if sources != authoritative or digest((pilot / "raw_source.jsonl").read_bytes()) != parent_manifest["source_sha256"]:
        raise ValueError("Production sources do not match the fingerprinted authoritative source")
    records = read_jsonl(production / "atoms.jsonl")
    failed = [r for r in records if r.record_id == TARGET]
    if len(failed) != 1 or len(records) != 5:
        raise ValueError("Unexpected pilot population")
    train = [r for r in records if r.split == "train"]
    if len(train) != 4 or not strict_report(train, {r.record_id: sources[r.record_id] for r in train})["accepted"]:
        raise ValueError("Original four training records must already pass")
    original = failed[0]
    raw_responses = [RawProviderResponse.model_validate_json(s) for s in
                     (production / "raw_responses.jsonl").read_text(encoding="utf-8").splitlines()]
    by_stage = {r.stage: r for r in raw_responses if r.record_id == TARGET}
    if len(by_stage) != 3:
        raise ValueError("Expected all three saved stages")
    record, probes, events, probe_xml, decomp_xml = prepare_record(original, sources[TARGET], by_stage)
    before = strict_report(records, sources)
    load_project_env()
    config = replace(MiniMaxConfig.from_env(), reasoning_effort="low",
                     max_completion_tokens=524288, timeout_seconds=1800)
    if config.model != "MiniMax-M3.1-Flash":
        raise ValueError("Repair must use the pilot model")
    out.mkdir(parents=True, exist_ok=False)
    save(out / "original_fingerprints.json", original_hashes)
    save(out / "validation.before.json", before)
    save(out / "transformations.json", events)
    (out / "recovered_probes.xml").write_text(probe_xml, encoding="utf-8")
    (out / "recovered_decomposition.xml").write_text(decomp_xml, encoding="utf-8")
    save(out / "prepared_validation_record.json", record.model_dump())
    shutil.copyfile(production / "raw_source.jsonl", out / "raw_source.jsonl")
    shutil.copyfile(production / "raw_responses.jsonl", out / "raw_responses.original.jsonl")
    retry_path = out / "raw_responses.retry.jsonl"
    retry_path.touch()
    save(out / "repair_manifest.json", {
        "created_at": datetime.now(timezone.utc).isoformat(), "parent_pilot": str(pilot),
        "record_id": TARGET, "model": config.model, "reasoning_effort": "low",
        "status": "prepared", "allowed_retry_stage": "question_annotation",
        "max_completion_tokens": config.max_completion_tokens, "max_retries": 0,
        "original_question_attempt": by_stage["question_annotation"].attempt,
        "scope": "One validation question-annotation stage; zero training/decomposition/probe calls",
    })
    original_source_questions = [q for q in original.qa_pairs if not q.is_generated_irrelevant]
    authoritative = build_authoritative_questions(original_source_questions, probes)
    atomizer = MiniMaxAtomizer(config)
    atomizer._begin_record(TARGET)
    atomizer._stage_attempts["question_annotation"] = by_stage["question_annotation"].attempt
    atomizer.response_sink = lambda r: write_raw_response_jsonl(r, retry_path)
    atom_xml, question_xml = atomizer._annotation_input_xml(record.atoms, authoritative)
    # The repaired span, not a rewritten answer, is supplied as read-only evidence.
    atom_root = ET.fromstring(atom_xml)
    for node, atom in zip(atom_root, record.atoms, strict=True):
        ET.SubElement(node, "source_span").text = atom.source_span
    system = QUESTION_SYSTEM + "\nUse the defeat-event atom for the winner question, not merely conference-champion facts."
    user = "\n".join((ET.tostring(atom_root, encoding="unicode"), question_xml,
                       atomizer._document_xml(sources[TARGET]["context"])))
    save(out / "retry_request.json", {"stage": "question_annotation", "attempt": 2,
                                      "system": system, "user": user, "reasoning_effort": "low"})
    print("Retrying validation question_annotation only (attempt 2, effort low)", flush=True)
    xml = atomizer._call("question_annotation", system, user)
    roles, notices = parse_question_roles_xml(xml)
    if notices:
        raise ValueError("Retried annotation has parse notices; artifacts retained, no auto-retry")
    known_ids = {atom.atom_id for atom in record.atoms}
    for role in roles:
        gold, support, distract = (set(role.gold_atom_ids), set(role.supporting_atom_ids),
                                   set(role.distractor_atom_ids))
        if (gold & support or gold & distract or support & distract
                or (gold | support | distract) - known_ids
                or any(len(values) != len(set(values)) for values in
                       (role.gold_atom_ids, role.supporting_atom_ids, role.distractor_atom_ids))):
            raise ValueError("Retried raw roles failed disjointness/reference checks; no auto-retry")
    record.qa_pairs = join_question_roles(authoritative, roles)
    events.append({"code": "RETRY_QUESTION_ANNOTATION", "stage": "question_annotation",
                   "before": [q.model_dump() for q in original.qa_pairs],
                   "after": [q.model_dump() for q in record.qa_pairs],
                   "attempt": 2, "reason": "Corrected bank and recovered probes require refreshed roles"})
    record.annotation_meta.irrelevant_parse_status = "ok"
    events.append({"code": "RESOLVE_PROBE_PARSE_STATUS", "stage": "pilot_repair",
                   "before": original.annotation_meta.irrelevant_parse_status, "after": "ok"})
    for event in events:
        record.annotation_meta.transformations.append(TransformationEvent(
            code=event["code"], stage=event["stage"], message=event.get("reason", event["code"]),
            atom_id=event.get("atom_id"), details=event))
    # Historical warnings/provenance stay intact; new changes carry separate structured events.
    record.annotation_meta.warnings.append("Pilot repair: historical parser warnings resolved; see transformations")
    validate_record(record, fuzzy_whitespace=False)
    new_records = [record if r.record_id == TARGET else r for r in records]
    # Preserve original serialized lines for all four training records byte-for-byte.
    lines = (production / "atoms.jsonl").read_bytes().splitlines(keepends=True)
    combined = b"".join((record.model_dump_json() + "\n").encode("utf-8")
                        if json.loads(line)["record_id"] == TARGET else line for line in lines)
    (out / "atoms.jsonl").write_bytes(combined)
    (out / "raw_responses.jsonl").write_bytes((production / "raw_responses.jsonl").read_bytes()
                                              + retry_path.read_bytes())
    save(out / "transformations.json", [event.model_dump() for event in record.annotation_meta.transformations])
    after = strict_report(new_records, sources)
    save(out / "validation.after.json", after)
    for split in ("train", "validation"):
        folder = out / split
        folder.mkdir()
        ids = {r.record_id for r in new_records if r.split == split}
        for name, key in (("atoms.jsonl", "record_id"), ("raw_source.jsonl", "id"),
                          ("raw_responses.jsonl", "record_id")):
            data = b"".join(line for line in (out / name).read_bytes().splitlines(keepends=True)
                            if str(json.loads(line)[key]) in ids)
            (folder / name).write_bytes(data)
        save(folder / "validation.json", strict_report([r for r in new_records if r.record_id in ids],
                                                       {key: sources[key] for key in ids}))
    unchanged = snapshot(pilot) == original_hashes
    train_unchanged = ((out / "train/atoms.jsonl").read_bytes() == (production / "train/atoms.jsonl").read_bytes())
    benchmark_usage = {effort: usage(pilot / "benchmark" / effort / "raw_responses.jsonl")
                       for effort in ("low", "medium", "high", "xhigh", "max")}
    save(out / "usage.json", {"benchmarks": benchmark_usage,
         "original_production": usage(production / "raw_responses.jsonl"),
         "repair_incremental": usage(retry_path), "production_with_retry": usage(out / "raw_responses.jsonl"),
         "incremental_cost_usd": None,
         "cost_note": "API responses include usage but no billed USD; no account billing data available."})
    manifest_path = out / "repair_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "validated" if after["accepted"] and unchanged and train_unchanged else "validation_failed"
    save(manifest_path, manifest)
    fingerprints = snapshot(out)
    save(out / "fingerprints.json", fingerprints)
    release = {"training_ready": after["accepted"] and unchanged and train_unchanged,
               "original_pilot_unchanged": unchanged, "training_records_byte_identical": train_unchanged,
               "reasoning_effort": "low", "provisional_default": True,
               "model": config.model, "before_errors": before["errors"], "after_errors": after["errors"],
               "fingerprints": {name: fingerprints[name] for name in ("atoms.jsonl", "raw_source.jsonl", "raw_responses.jsonl")},
               "new_api_calls": 1, "training_started": False}
    save(out / "release.json", release)
    print(json.dumps(release, indent=2), flush=True)
    if not release["training_ready"]:
        raise SystemExit("Strict validation failed; inspect saved repair, do not train")


if __name__ == "__main__":
    main()

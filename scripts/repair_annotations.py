"""Repair audited annotations into a separate bundle; preserve all valid cached records.

No source answers are edited. --allow-api authorizes one question-stage attempt
per record whose parse status failed, not full-record regeneration or auto-retries.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

from doc_to_atom.atomization.client import MiniMaxAtomizer, MiniMaxConfig
from doc_to_atom.atomization.models import Atom, QuestionAnnotation, TransformationEvent
from doc_to_atom.atomization.prompts import QUESTION_SYSTEM
from doc_to_atom.atomization.parser import parse_question_roles_xml
from doc_to_atom.atomization.provenance import RawProviderResponse, write_raw_response_jsonl
from doc_to_atom.atomization.question_join import join_question_roles
from doc_to_atom.atomization.validation import source_span_matches, validate_record
from doc_to_atom.atomization.xml_recovery import recover_annotation_xml
from doc_to_atom.config import load_project_env
from doc_to_atom.data.io import read_jsonl, validate_jsonl
from doc_to_atom.data.semantic_validation import audit_records, load_source_records


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def log(record, code, reason, before, after, atom_id=None):
    record.annotation_meta.transformations.append(TransformationEvent(
        code=code, stage="source_reviewed_repair", message=reason, atom_id=atom_id,
        details={"before": before, "after": after}))


def repair_spans(record, replacements):
    for atom in record.atoms:
        if source_span_matches(atom.source_span, record.context):
            continue
        updated = atom.source_span
        for old, new in replacements:
            updated = updated.replace(old, new)
        # No approximate match, case folding, or fabricated context is allowed.
        if updated == atom.source_span or updated not in record.context:
            raise ValueError(f"Unreviewed source span: {record.record_id}/{atom.atom_id}")
        log(record, "RESTORE_VERBATIM_SOURCE_SPAN", "Copy source spelling/grammar exactly",
            atom.source_span, updated, atom.atom_id)
        atom.source_span = updated
        atom.span_valid = True


def apply_plan(record, plan):
    for patch in plan["atoms"]:
        if patch["record_id"] != record.record_id:
            continue
        atom = next(a for a in record.atoms if a.atom_id == patch["atom_id"])
        before = atom.model_dump()
        if "content" in patch:
            atom.content = patch["content"]
        if "content_prefix" in patch:
            atom.content = patch["content_prefix"] + atom.content[0].lower() + atom.content[1:]
        if "source_span" in patch:
            atom.source_span = patch["source_span"]
        atom.source_span = patch.get("span_prefix", "") + atom.source_span + patch.get("span_suffix", "")
        if atom.source_span not in record.context:
            raise ValueError(f"Plan span not in authoritative context: {record.record_id}/{atom.atom_id}")
        log(record, "SOURCE_REVIEWED_ATOM_CORRECTION", patch["reason"], before, atom.model_dump(), atom.atom_id)
    for patch in plan["new_atoms"]:
        if patch["record_id"] != record.record_id:
            continue
        atom = Atom(**{k: v for k, v in patch.items() if k not in {"record_id", "reason"}},
                    is_answer_bearing=True, abstraction_level="evidence", confidence=1.0)
        if atom.source_span not in record.context or atom.atom_id in {a.atom_id for a in record.atoms}:
            raise ValueError("New reviewed atom must have a unique ID and a verbatim source span")
        record.atoms.append(atom)
        log(record, "ADD_OMITTED_EXPLICIT_SOURCE_FACT", patch["reason"], None, atom.model_dump(), atom.atom_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--raw-responses", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--plan", type=Path, default=Path("configs/annotation_repairs/squad_medium_20261104.json"))
    parser.add_argument("--allow-api", action="store_true")
    args = parser.parse_args()
    out = args.output_dir
    if out.exists():
        raise FileExistsError("Refusing to overwrite an existing repair bundle or repeat paid requests")
    originals = {str(p): sha(p) for p in (args.input, args.source, args.raw_responses)}
    sources = load_source_records(args.source)
    records = read_jsonl(args.input)
    baseline = audit_records(records, sources)
    affected = {issue["record_id"] for issue in baseline["issues"]}
    affected.update(r.record_id for r in records
                    if any(not source_span_matches(a.source_span, r.context) for a in r.atoms))
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    raw = [RawProviderResponse.model_validate_json(s) for s in
           args.raw_responses.read_text(encoding="utf-8").splitlines() if s.strip()]
    latest = {(r.record_id, r.stage): r for r in raw}
    retries = [r for r in records if r.record_id in affected and
               (r.annotation_meta.question_parse_status != "ok" or r.annotation_meta.irrelevant_parse_status != "ok")]
    if retries and not args.allow_api:
        raise ValueError(f"{len(retries)} question-stage retries require explicit --allow-api")
    out.mkdir(parents=True)
    save(out / "original_fingerprints.json", originals)
    save(out / "validation.before.json", {"semantic": baseline, "stats": validate_jsonl(args.input, source_path=args.source)})
    shutil.copyfile(args.source, out / "raw_source.jsonl")
    shutil.copyfile(args.raw_responses, out / "raw_responses.jsonl")
    shutil.copyfile(args.plan, out / "repair_plan.json")
    save(out / "manifest.json", {"status": "in_progress", "affected_records": sorted(affected),
         "retry_record_ids": [r.record_id for r in retries], "model": "MiniMax-M3.1-Flash",
         "reasoning_effort": "low", "max_completion_tokens": 524288,
         "method": "Source-reviewed corrections plus exactly one question-stage retry per failed record"})
    config = None
    if retries:
        load_project_env()
        config = replace(MiniMaxConfig.from_env(), reasoning_effort="low", max_completion_tokens=524288,
                         timeout_seconds=1800)
    retry_ids = {r.record_id for r in retries}
    new_raw = out / "raw_responses.retry.jsonl"
    new_raw.touch()
    repaired = {}
    for record in records:
        if record.record_id not in affected:
            continue
        source = sources[record.record_id]
        if record.context != source["context"]:
            raise ValueError("Atomized context differs from authoritative source")
        before = record.model_dump()
        repair_spans(record, plan["span_typo_replacements"])
        apply_plan(record, plan)
        if record.record_id in retry_ids:
            probe_raw = latest[(record.record_id, "irrelevant_probe")]
            derived, edits = recover_annotation_xml(probe_raw.content, "irrelevant_questions")
            log(record, "RECOVER_SAVED_PROBE_XML", "Syntax-only edits; original raw response unchanged",
                {"content_sha256": probe_raw.sha256, "content": probe_raw.content},
                {"content": derived, "edits": edits})
            probes = [QuestionAnnotation(
                question_id=f"generated_irrelevant_{i}", question=node.text.strip(), question_type="irrelevant",
                is_irrelevant=True, is_generated_irrelevant=True, answer_provenance="generated_irrelevant")
                for i, node in enumerate(ET.fromstring(derived))]
            questions = [QuestionAnnotation(
                question_id=str(q.get("id", q.get("question_id"))), question=q["question"],
                answers=q.get("answers", []), answer_provenance="source", question_type=q.get("question_type", "single_hop"))
                for q in source["questions"]] + probes
            client = MiniMaxAtomizer(config)
            client._begin_record(record.record_id)
            attempt = max([r.attempt for r in raw if r.record_id == record.record_id and r.stage == "question_annotation"] + [0])
            client._stage_attempts["question_annotation"] = attempt
            client.response_sink = lambda response: write_raw_response_jsonl(response, new_raw)
            atoms_xml, questions_xml = client._annotation_input_xml(record.atoms, tuple(questions))
            bank = ET.fromstring(atoms_xml)
            for node, atom in zip(bank, record.atoms, strict=True):
                ET.SubElement(node, "source_span").text = atom.source_span
            system = QUESTION_SYSTEM + "\nUse both atom content and source_span. Original source questions are answerable; do not relabel them irrelevant."
            user = ET.tostring(bank, encoding="unicode") + "\n" + questions_xml + "\n" + client._document_xml(source["context"])
            save(out / f"request-{record.record_id}.json", {"system": system, "user": user, "attempt": attempt + 1})
            print(f"Repairing question_annotation only: {record.record_id}", flush=True)
            response = client._call("question_annotation", system, user)
            response, role_edits = recover_annotation_xml(response, "question_annotations")
            roles, notices = parse_question_roles_xml(response)
            if notices:
                raise ValueError("Retried roles have parse errors; no automatic retry")
            known = {a.atom_id for a in record.atoms}
            for role in roles:
                gold, support, distract = map(set, (role.gold_atom_ids, role.supporting_atom_ids, role.distractor_atom_ids))
                if gold & support or gold & distract or support & distract or (gold | support | distract) - known:
                    raise ValueError("Retried raw roles violate disjointness/references")
            record.qa_pairs = join_question_roles(questions, roles)
            log(record, "RETRY_FAILED_QUESTION_STAGE", "Corrected/recovered bank and authoritative source question pool",
                before["qa_pairs"], {"qa_pairs": [q.model_dump() for q in record.qa_pairs],
                                     "xml_edits": role_edits, "attempt": attempt + 1})
            record.annotation_meta.question_parse_status = "ok"
            record.annotation_meta.irrelevant_parse_status = "ok"
            record.annotation_meta.errors = []
        validate_record(record)
        save(out / f"repair-{record.record_id}.json", {"before": before, "after": record.model_dump()})
        repaired[record.record_id] = record
    # Preserve every unaffected serialized record exactly, not just its semantic fields.
    original_lines = args.input.read_bytes().splitlines(keepends=True)
    (out / "atoms.jsonl").write_bytes(b"".join(
        (repaired[json.loads(line)["record_id"]].model_dump_json() + "\n").encode("utf-8")
        if json.loads(line)["record_id"] in repaired else line for line in original_lines if line.strip()))
    with (out / "raw_responses.jsonl").open("ab") as sink:
        sink.write(new_raw.read_bytes())
    report = audit_records(read_jsonl(out / "atoms.jsonl"), sources)
    stats = validate_jsonl(out / "atoms.jsonl", source_path=out / "raw_source.jsonl")
    save(out / "validation.after.json", {"semantic": report, "stats": stats})
    unchanged = all(sha(path) == value for path, value in originals.items())
    accepted = (not report["errors"] and not stats["invalid_spans"]
                and report["answer_preservation_rate"] == 1.0 and len(records) == len(sources) and unchanged)
    retry_rows = [RawProviderResponse.model_validate_json(s) for s in new_raw.read_text(encoding="utf-8").splitlines()]
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "validated" if accepted else "validation_failed"
    save(manifest_path, manifest)
    save(out / "release.json", {"training_ready": accepted, "source_artifacts_unchanged": unchanged,
         "repaired_records": len(repaired), "unchanged_records": len(records) - len(repaired),
         "new_api_calls": len(retry_rows), "new_total_tokens": sum(
             json.loads(r.raw_body)["usage"]["total_tokens"] for r in retry_rows), "training_started": False,
         "fingerprints": {name: sha(out / name) for name in ("atoms.jsonl", "raw_source.jsonl", "raw_responses.jsonl")}})
    print(json.dumps({"output": str(out), "accepted": accepted, "errors": report["errors"],
                      "invalid_spans": stats["invalid_spans"], "source_answers_preserved": report["source_answer_matches"]}, indent=2))
    if not accepted:
        raise SystemExit("Repair retained for inspection; source-aware validation failed. Do not train.")


if __name__ == "__main__":
    main()

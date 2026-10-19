from __future__ import annotations

import argparse
import json
import re
import statistics
import time
import unicodedata
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.nn import functional as F

from doc_to_atom.config import load_config, load_project_env
from doc_to_atom.data.io import read_jsonl, validate_jsonl
from doc_to_atom.data.semantic_validation import audit_records
from doc_to_atom.data.targets import resolve_target
from doc_to_atom.evaluation.metrics import rouge_l, word_f1
from doc_to_atom.training.checkpoint import load_checkpoint
from doc_to_atom.training.factory import build_system
from doc_to_atom.training.loop import _answer_logits, _causal_example, _memory, _tokenize

MODES = (
    "base_query_only",
    "base_full_context",
    "doc2atom_predicted",
    "doc2atom_forced_gold",
    "base_gold_atom_context",
)


def _answer_text(answer: str | list[str] | None) -> str:
    if isinstance(answer, list):
        return str(answer[0]) if answer else ""
    return str(answer or "")


def _normalized(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return " ".join(re.findall(r"\w+", text, flags=re.UNICODE))


def _prompt(mode: str, question: str, document: str, gold_text: str) -> str:
    if mode == "base_full_context":
        return f"Document:\n{document}\n\nQuestion: {question}\nAnswer:"
    if mode == "base_gold_atom_context":
        return f"Gold atoms:\n{gold_text}\n\nQuestion: {question}\nAnswer:"
    return f"Question: {question}\nAnswer:"


def _generate(model, tokenizer, prompt: str, max_new_tokens: int, injector, composed):
    inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
    activation = nullcontext() if composed is None else injector.activate(composed)
    with activation:
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )
    return tokenizer.decode(
        output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()


def _teacher_forced_logits(
    model,
    tokenizer,
    prompt: str,
    answer: str,
    max_length: int,
    injector,
    composed,
):
    device = next(model.parameters()).device
    inputs, labels, answer_count = _causal_example(tokenizer, prompt, answer, max_length, device)
    activation = nullcontext() if composed is None else injector.activate(composed)
    with activation:
        logits = model(**inputs, use_cache=False).logits
    answer_logits = _answer_logits(logits, answer_count).float()
    answer_ids = labels[labels != -100][-answer_count:]
    nll = F.cross_entropy(answer_logits.reshape(-1, answer_logits.size(-1)), answer_ids)
    return answer_logits, float(nll)


def _kl(student: torch.Tensor, teacher: torch.Tensor, temperature: float) -> float:
    common = min(student.shape[1], teacher.shape[1])
    value = F.kl_div(
        F.log_softmax(student[:, -common:] / temperature, dim=-1),
        F.softmax(teacher[:, -common:] / temperature, dim=-1),
        reduction="batchmean",
    )
    return float(value)


def _mean(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row[key] is not None]
    return statistics.fmean(values) if values else None


def _summaries(rows: list[dict]) -> dict:
    result = {}
    for split in ("validation", "test"):
        result[split] = {}
        for mode in MODES:
            selected = [row for row in rows if row["split"] == split and row["mode"] == mode]
            result[split][mode] = {
                "count": len(selected),
                "word_f1": _mean(selected, "word_f1"),
                "rouge_l": _mean(selected, "rouge_l"),
                "exact_match_rate": _mean(selected, "exact_match"),
                "near_match_rate": _mean(selected, "near_match"),
                "answer_containment_rate": _mean(selected, "answer_containment"),
                "exact_or_near_rate": _mean(selected, "exact_or_near"),
                "teacher_forced_nll": _mean(selected, "teacher_forced_nll"),
                "teacher_student_kl": _mean(selected, "teacher_student_kl"),
                "adapter_norm": _mean(selected, "adapter_norm"),
            }
    return result


@torch.inference_mode()
def main() -> None:
    load_project_env()
    parser = argparse.ArgumentParser(description="Held-out component-isolation evaluation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "results.json").exists():
        raise FileExistsError(f"refusing to overwrite completed run: {output}")

    config = load_config(args.config)
    data_stats = validate_jsonl(args.data, reject_span_mismatch=True)
    records = read_jsonl(args.data)
    semantic = audit_records(records)
    if semantic["errors"] or semantic["warnings"]:
        raise ValueError(f"held-out input is not semantically clean: {semantic}")
    heldout = [record for record in records if record.split in {"validation", "test"}]
    if {record.split for record in heldout} != {"validation", "test"}:
        raise ValueError("both validation and test records are required")

    system, tokenizer, _targets = build_system(config)
    step = load_checkpoint(
        {
            "encoder_projection": system.encoder.projection,
            "compiler": system.compiler,
            "router": system.router,
        },
        args.checkpoint,
    )
    system.eval()
    device = next(system.parameters()).device
    temperature = config["training"]["distillation_temperature"]
    rows = []
    started = time.perf_counter()

    # Validation is evaluated first. Test remains measurement-only and changes no state.
    for split in ("validation", "test"):
        for record in [item for item in heldout if item.split == split]:
            bank = _memory(system, tokenizer, record, config)
            atom_index = {atom.atom_id: index for index, atom in enumerate(record.atoms)}
            for qa in [item for item in record.qa_pairs if not item.is_irrelevant]:
                answer = resolve_target(qa).text
                gold_indices = torch.tensor(
                    [atom_index[atom_id] for atom_id in qa.gold_atom_ids], device=device
                )
                forced_weights = torch.full(
                    (len(gold_indices),),
                    1.0 / len(gold_indices),
                    device=device,
                    dtype=bank.keys.dtype,
                )
                forced = system.composer(bank, gold_indices, forced_weights)

                query_inputs = _tokenize(tokenizer, [qa.question], 256, device)
                query_embedding = system.encoder.encode_queries(**query_inputs)
                routed = system.router(query_embedding, bank.keys, bank.atoms or [], qa.question, 8)
                predicted = system.composer(bank, routed.selected_indices, routed.routing_weights)
                predicted_ids = [
                    record.atoms[index].atom_id for index in routed.selected_indices.tolist()
                ]
                gold_text = "\n".join(
                    record.atoms[index].content for index in gold_indices.tolist()
                )
                teacher_prompt = _prompt(
                    "base_full_context", qa.question, record.context, gold_text
                )
                teacher_logits, _teacher_nll = _teacher_forced_logits(
                    system.base_model,
                    tokenizer,
                    teacher_prompt,
                    answer,
                    config["data"]["max_context_tokens"],
                    system.injector,
                    None,
                )

                mode_state = {
                    "base_query_only": (None, [], []),
                    "base_full_context": (None, [], []),
                    "doc2atom_predicted": (
                        predicted,
                        predicted_ids,
                        [float(weight) for weight in routed.routing_weights],
                    ),
                    "doc2atom_forced_gold": (
                        forced,
                        list(qa.gold_atom_ids),
                        [float(weight) for weight in forced_weights],
                    ),
                    "base_gold_atom_context": (None, list(qa.gold_atom_ids), []),
                }
                for mode in MODES:
                    composed, selected_ids, weights = mode_state[mode]
                    prompt = _prompt(mode, qa.question, record.context, gold_text)
                    prediction = _generate(
                        system.base_model,
                        tokenizer,
                        prompt,
                        config["evaluation"]["max_new_tokens"],
                        system.injector,
                        composed,
                    )
                    logits, nll = _teacher_forced_logits(
                        system.base_model,
                        tokenizer,
                        prompt,
                        answer,
                        config["data"]["max_context_tokens"],
                        system.injector,
                        composed,
                    )
                    f1 = word_f1(prediction, answer)
                    normalized_prediction = _normalized(prediction)
                    normalized_answer = _normalized(answer)
                    containment = float(
                        bool(normalized_answer) and normalized_answer in normalized_prediction
                    )
                    exact = float(normalized_prediction == normalized_answer)
                    near = float(not exact and (containment or f1 >= 0.8))
                    rows.append(
                        {
                            "split": split,
                            "record_id": record.record_id,
                            "question_id": qa.question_id,
                            "question": qa.question,
                            "answer": answer,
                            "mode": mode,
                            "prediction": prediction,
                            "word_f1": f1,
                            "rouge_l": rouge_l(prediction, answer),
                            "exact_match": exact,
                            "near_match": near,
                            "answer_containment": containment,
                            "exact_or_near": max(exact, near),
                            "teacher_forced_nll": nll,
                            "teacher_student_kl": _kl(logits, teacher_logits, temperature),
                            "adapter_norm": (
                                None if composed is None else float(composed.adapter_norm())
                            ),
                            "selected_atom_ids": selected_ids,
                            "routing_weights": weights,
                        }
                    )

    summaries = _summaries(rows)
    gaps = {}
    for split in ("validation", "test"):
        split_summary = summaries[split]
        gaps[split] = {
            "routing_gap_word_f1": (
                split_summary["doc2atom_forced_gold"]["word_f1"]
                - split_summary["doc2atom_predicted"]["word_f1"]
            ),
            "routing_gap_nll": (
                split_summary["doc2atom_predicted"]["teacher_forced_nll"]
                - split_summary["doc2atom_forced_gold"]["teacher_forced_nll"]
            ),
            "parametric_memory_gap_word_f1": (
                split_summary["base_gold_atom_context"]["word_f1"]
                - split_summary["doc2atom_forced_gold"]["word_f1"]
            ),
            "parametric_memory_gap_nll": (
                split_summary["doc2atom_forced_gold"]["teacher_forced_nll"]
                - split_summary["base_gold_atom_context"]["teacher_forced_nll"]
            ),
        }

    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_step": step,
        "data": data_stats,
        "heldout_documents": len(heldout),
        "answerable_examples": len(rows) // len(MODES),
        "forced_gold_weight_strategy": "uniform over exact gold_atom_ids; weights sum to 1",
        "near_match_definition": (
            "normalized non-exact prediction containing the normalized gold answer, "
            "or with word F1 >= 0.8"
        ),
        "teacher_student_kl": (
            "KL(teacher full-document distribution || mode distribution), temperature=1.5, "
            "matching the training implementation without T^2 rescaling"
        ),
        "test_policy": "evaluated after validation; no training or tuning occurs in this script",
        "summaries": summaries,
        "gaps": gaps,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    (output / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

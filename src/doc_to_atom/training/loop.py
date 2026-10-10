from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import get_cosine_schedule_with_warmup

from doc_to_atom.atomization.models import StructuredRecord
from doc_to_atom.compiler.memory import MemoryBank
from doc_to_atom.losses.objectives import (
    composition_consistency_loss,
    confidence_alignment_loss,
    conflict_listwise_loss,
    irrelevant_suppression_loss,
    listwise_rerank_loss,
    protection_loss,
    routing_loss,
    sparse_regularization_loss,
)
from doc_to_atom.training.checkpoint import save_checkpoint
from doc_to_atom.training.curriculum import stage_for_step
from doc_to_atom.training.trainer import MultiObjectiveTrainer

LOGGER = logging.getLogger(__name__)


def _model_device(model) -> torch.device:
    return next(model.parameters()).device


def _tokenize(tokenizer, texts: list[str], max_length: int, device: torch.device):
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {key: value.to(device) for key, value in encoded.items()}


def _causal_example(tokenizer, prompt: str, answer: str, max_length: int, device: torch.device):
    prompt_ids = tokenizer(prompt, add_special_tokens=True, truncation=False)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    room = max(1, max_length - len(answer_ids))
    prompt_ids = prompt_ids[-room:]
    ids = (prompt_ids + answer_ids)[:max_length]
    answer_count = min(len(answer_ids), len(ids))
    input_ids = torch.tensor([ids], device=device)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    labels[:, : len(ids) - answer_count] = -100
    return {"input_ids": input_ids, "attention_mask": attention_mask}, labels, answer_count


def _answer_logits(logits: torch.Tensor, answer_count: int) -> torch.Tensor:
    # Positions immediately before each answer token predict the answer sequence.
    return logits[:, -(answer_count + 1) : -1, :]


def _answer_text(answer) -> str:
    if isinstance(answer, list):
        return str(answer[0]) if answer else ""
    return str(answer or "")


def _memory(system, tokenizer, record: StructuredRecord, config: dict) -> MemoryBank:
    device = _model_device(system.base_model)
    atom_inputs = _tokenize(
        tokenizer,
        [atom.content for atom in record.atoms],
        config["data"]["max_atom_tokens"],
        device,
    )
    return system.build_memory(atom_inputs, record.atoms)


def _one_question(system, tokenizer, record, qa, bank, config, step: int):
    device = _model_device(system.base_model)
    stage = stage_for_step(step)
    answer = "unanswerable" if qa.is_irrelevant else _answer_text(qa.answer)
    if not answer:
        raise ValueError(f"{record.record_id}/{qa.question_id}: missing answer")

    # PAPER_UNSPECIFIED prompt literals; intentionally plain and documented.
    teacher_prompt = f"Document:\n{record.context}\n\nQuestion: {qa.question}\nAnswer:"
    student_prompt = f"Question: {qa.question}\nAnswer:"
    teacher_inputs, _teacher_labels, teacher_count = _causal_example(
        tokenizer, teacher_prompt, answer, config["data"]["max_context_tokens"], device
    )
    student_inputs, labels, answer_count = _causal_example(
        tokenizer,
        student_prompt,
        answer,
        config["data"]["max_answer_tokens"] + 256,
        device,
    )
    query_inputs = _tokenize(tokenizer, [qa.question], 256, device)
    teacher_logits = system.teacher_forward(teacher_inputs)
    student = system.student_forward(student_inputs, query_inputs, qa.question, bank, stage.top_k)
    student_answer = _answer_logits(student.logits, answer_count)
    teacher_answer = _answer_logits(teacher_logits, min(teacher_count, answer_count))
    common = min(student_answer.shape[1], teacher_answer.shape[1])
    student_answer, teacher_answer = student_answer[:, -common:], teacher_answer[:, -common:]

    answer_ids = labels[labels != -100][-common:]
    lm = F.cross_entropy(student_answer.reshape(-1, student_answer.size(-1)), answer_ids)
    distill = F.kl_div(
        F.log_softmax(student_answer / config["training"]["distillation_temperature"], -1),
        F.softmax(teacher_answer / config["training"]["distillation_temperature"], -1),
        reduction="batchmean",
    )

    all_ids = [atom.atom_id for atom in record.atoms]
    relevant = set(qa.gold_atom_ids + qa.supporting_atom_ids)  # Assumption documented.
    gold_mask = torch.tensor([atom_id in relevant for atom_id in all_ids], device=device)
    distractor_mask = torch.tensor(
        [atom_id in qa.distractor_atom_ids for atom_id in all_ids], device=device
    )
    route = routing_loss(student.router.all_stage1_scores, gold_mask, distractor_mask)

    candidate_gold = gold_mask[student.router.stage1_indices]
    rerank = listwise_rerank_loss(
        student.router.reranker_scores + student.router.metadata_biases, candidate_gold
    )

    if relevant:
        forced_indices = torch.tensor(
            [index for index, atom_id in enumerate(all_ids) if atom_id in relevant], device=device
        )
        forced_weights = F.softmax(student.router.all_stage1_scores[forced_indices], dim=0)
        forced = system.composer(bank, forced_indices, forced_weights)
        with system.injector.activate(forced):
            forced_logits = system.base_model(**student_inputs, use_cache=False).logits
        composition = composition_consistency_loss(
            student_answer, _answer_logits(forced_logits, answer_count)[:, -common:]
        )
    else:
        composition = student_answer.sum() * 0

    selected_atoms = [record.atoms[index] for index in student.router.selected_indices.tolist()]
    selected_confidences = torch.tensor([atom.confidence for atom in selected_atoms], device=device)
    gate_tensor = torch.stack(list(student.composed.gates.values()))
    losses = {
        "lm": lm,
        "distill": distill,
        "route": route,
        "rerank": rerank,
        "irrelevant": irrelevant_suppression_loss(student.composed)
        if qa.is_irrelevant
        else student_answer.sum() * 0,
        "protect": protection_loss(student.composed),
        "sparse": sparse_regularization_loss(
            gate_tensor, config["training"]["sparse_target_density"]
        ),
        "composition": composition,
        "conflict": conflict_listwise_loss(
            student.router.all_stage1_scores,
            [atom.conflict_group for atom in record.atoms],
            gold_mask,
        ),
        "confidence": confidence_alignment_loss(
            student.router.routing_weights, selected_confidences
        ),
    }
    bundle = MultiObjectiveTrainer(config).combine(step, losses)
    trace = {
        "losses": {name: float(value.detach()) for name, value in losses.items()},
        "selected_atom_ids": [atom.atom_id for atom in selected_atoms],
        "routing_weights": [float(value) for value in student.router.routing_weights.detach()],
        "routing_weight_sum": float(student.router.routing_weights.detach().sum()),
        "stage1_candidate_count": int(student.router.stage1_indices.numel()),
        "final_top_k": int(student.router.selected_indices.numel()),
    }
    return bundle, float(lm.detach()), trace


def run_training(
    system, tokenizer, records: list[StructuredRecord], config: dict, output_dir: Path
):
    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    train_records = [record for record in records if record.split not in {"validation", "test"}]
    validation_records = [record for record in records if record.split == "validation"]
    if not train_records:
        raise ValueError("no training records after split filtering")
    trainable = [parameter for parameter in system.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config["training"]["learning_rate"],
        betas=tuple(config["training"]["betas"]),
        weight_decay=config["training"]["weight_decay"],
    )
    total_steps = max(1, len(train_records) * config["training"]["max_epochs"])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, config["training"]["warmup_steps"], total_steps
    )
    accumulation = config["training"].get("gradient_accumulation_steps", 1)
    global_step = 0
    best_validation = float("inf")
    bad_epochs = 0
    last_trace = None
    log_path = output_dir / "train_log.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(config["training"]["max_epochs"]):
        system.train()
        # Keep the base model frozen and deterministic while trainable heads remain in train mode.
        system.base_model.eval()
        for record in train_records:
            bank = _memory(system, tokenizer, record, config)
            record_totals = []
            record_lm = []
            for qa in record.qa_pairs[: config["data"]["max_questions_per_sample"]]:
                bundle, lm_value, trace = _one_question(
                    system, tokenizer, record, qa, bank, config, global_step
                )
                record_totals.append(bundle.total)
                record_lm.append(lm_value)
                last_trace = trace
            if not record_totals:
                continue
            record_total = torch.stack(record_totals).mean()
            (record_total / accumulation).backward()
            if (global_step + 1) % accumulation == 0:
                torch.nn.utils.clip_grad_norm_(trainable, config["training"]["gradient_clip"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "step": global_step,
                            "lm": sum(record_lm) / len(record_lm),
                            "total": float(record_total.detach()),
                        }
                    )
                    + "\n"
                )
            global_step += 1
            max_steps = config["training"].get("max_steps")
            if max_steps is not None and global_step >= max_steps:
                break

        if global_step % accumulation != 0:
            torch.nn.utils.clip_grad_norm_(trainable, config["training"]["gradient_clip"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        validation_lm = _validation_lm(system, tokenizer, validation_records, config, global_step)
        save_checkpoint(
            {
                "encoder_projection": system.encoder.projection,
                "compiler": system.compiler,
                "router": system.router,
            },
            output_dir / "last.pt",
            optimizer,
            global_step,
        )
        if validation_records:
            if validation_lm < best_validation:
                best_validation = validation_lm
                bad_epochs = 0
                save_checkpoint(
                    {
                        "encoder_projection": system.encoder.projection,
                        "compiler": system.compiler,
                        "router": system.router,
                    },
                    output_dir / "best.pt",
                    optimizer,
                    global_step,
                )
            else:
                bad_epochs += 1
                patience = config["training"].get("early_stopping_patience")
                if patience is None:
                    raise ValueError("PAPER_UNSPECIFIED early-stopping patience")
                if bad_epochs >= patience:
                    LOGGER.info("early stopping at epoch %s", epoch)
                    break
        max_steps = config["training"].get("max_steps")
        if max_steps is not None and global_step >= max_steps:
            break
    return {
        "steps": global_step,
        "best_validation_lm": best_validation,
        "last_trace": last_trace,
        "elapsed_seconds": time.perf_counter() - started,
        "cuda_available": torch.cuda.is_available(),
        "cuda_peak_allocated_bytes": (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        ),
        "cuda_peak_reserved_bytes": (
            torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0
        ),
    }


@torch.no_grad()
def _validation_lm(system, tokenizer, records, config, step):
    if not records:
        return float("nan")
    system.eval()
    values = []
    for record in records:
        bank = _memory(system, tokenizer, record, config)
        for qa in record.qa_pairs[: config["data"]["max_questions_per_sample"]]:
            _bundle, lm, _trace = _one_question(system, tokenizer, record, qa, bank, config, step)
            values.append(lm)
    return sum(values) / len(values)

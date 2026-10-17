from __future__ import annotations

import json
import logging
import math
import random
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import get_cosine_schedule_with_warmup

from doc_to_atom.atomization.models import StructuredRecord
from doc_to_atom.compiler.memory import MemoryBank
from doc_to_atom.config import config_fingerprint
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
from doc_to_atom.runtime.run import atomic_write_json
from doc_to_atom.training.checkpoint import load_training_checkpoint, save_training_checkpoint
from doc_to_atom.training.curriculum import stage_for_step
from doc_to_atom.training.state import TrainerState
from doc_to_atom.training.trainer import EarlyStopping, MultiObjectiveTrainer

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


def _decode(tokenizer, token_ids: list[int]) -> str:
    if hasattr(tokenizer, "decode"):
        return tokenizer.decode(token_ids, skip_special_tokens=True)
    return " ".join(map(str, token_ids))


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _aggregate_traces(traces: list[dict]) -> dict:
    if not traces:
        return {"count": 0}
    loss_names = traces[0]["losses"]
    metric_names = (
        "total_loss",
        "gold_recall_at_1",
        "gold_recall_at_k",
        "mrr",
        "routing_entropy",
        "adapter_norm",
        "token_accuracy",
    )
    return {
        "count": len(traces),
        "losses": {
            name: _mean([trace["losses"][name] for trace in traces]) for name in loss_names
        },
        **{
            name: _mean(
                [float(trace[name]) for trace in traces if trace.get(name) is not None]
            )
            for name in metric_names
        },
        "examples": [
            {
                "question_id": trace["question_id"],
                "selected_atom_ids": trace["selected_atom_ids"],
                "routing_weights": trace["routing_weights"],
                "prediction": trace["prediction"],
                "target": trace["target"],
            }
            for trace in traces
        ],
    }


def _split_trace_report(traces: list[dict]) -> dict:
    return {
        "overall": _aggregate_traces(traces),
        "answerable": _aggregate_traces(
            [trace for trace in traces if trace["group"] == "answerable"]
        ),
        "generated_irrelevant": _aggregate_traces(
            [trace for trace in traces if trace["group"] == "generated_irrelevant"]
        ),
    }


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
    candidate_scores = (
        student.router.stage1_scores
        + config["router"]["rerank_fusion_weight"] * student.router.reranker_scores
        + student.router.metadata_biases
    )
    ranked_indices = student.router.stage1_indices[torch.argsort(candidate_scores, descending=True)]
    gold_indices = torch.where(gold_mask)[0]
    diagnostic_k = min(config.get("evaluation", {}).get("routing_recall_k", 4), len(all_ids))
    recall_at_1 = recall_at_k = mrr = None
    if gold_indices.numel():
        gold_set = set(gold_indices.tolist())
        ranked = ranked_indices.tolist()
        recall_at_1 = len(gold_set.intersection(ranked[:1])) / len(gold_set)
        recall_at_k = len(gold_set.intersection(ranked[:diagnostic_k])) / len(gold_set)
        first_rank = next((rank for rank, index in enumerate(ranked, 1) if index in gold_set), None)
        mrr = 0.0 if first_rank is None else 1.0 / first_rank
    weights = student.router.routing_weights.detach().float()
    predicted_ids = student_answer.detach().argmax(dim=-1)[0]
    trace = {
        "question_id": qa.question_id,
        "group": "generated_irrelevant" if qa.is_generated_irrelevant else "answerable",
        "total_loss": float(bundle.total.detach()),
        "losses": {name: float(value.detach()) for name, value in losses.items()},
        "selected_atom_ids": [atom.atom_id for atom in selected_atoms],
        "routing_weights": [float(value) for value in student.router.routing_weights.detach()],
        "routing_weight_sum": float(student.router.routing_weights.detach().sum()),
        "routing_entropy": float(-(weights * weights.clamp_min(1e-12).log()).sum()),
        "gold_recall_at_1": recall_at_1,
        "gold_recall_at_k": recall_at_k,
        "mrr": mrr,
        "adapter_norm": float(student.composed.adapter_norm().detach()),
        "token_accuracy": float((predicted_ids == answer_ids).float().mean()),
        "prediction": _decode(tokenizer, predicted_ids.tolist()),
        "target": answer,
        "stage1_candidate_count": int(student.router.stage1_indices.numel()),
        "final_top_k": int(student.router.selected_indices.numel()),
    }
    return bundle, float(lm.detach()), trace


def _trainable_modules(system) -> dict[str, torch.nn.Module]:
    return {
        "encoder_projection": system.encoder.projection,
        "compiler": system.compiler,
        "router": system.router,
    }


def _epoch_order(length: int, seed: int, epoch: int) -> list[int]:
    order = list(range(length))
    random.Random(seed + epoch).shuffle(order)
    return order


def _resume_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.suffix == ".json" and candidate.is_file():
        pointer = json.loads(candidate.read_text(encoding="utf-8"))
        return candidate.parent / pointer["path"]
    return candidate


def _metric_value(report: dict, path: str) -> float:
    current: object = report
    for component in path.split("."):
        if not isinstance(current, dict) or component not in current:
            raise ValueError(f"missing early-stopping metric: {path}")
        current = current[component]
    return float("nan") if current is None else float(current)


def run_training(
    system,
    tokenizer,
    records: list[StructuredRecord],
    config: dict,
    output_dir: Path,
    *,
    resume_from: str | Path | None = None,
    tracker=None,
    stop_after_optimizer_steps: int | None = None,
):
    """Train at document granularity; all schedules advance on optimizer updates."""
    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    train_records = [record for record in records if record.split not in {"validation", "test"}]
    validation_records = [record for record in records if record.split == "validation"]
    if not train_records:
        raise ValueError("no training records after split filtering")
    training_config = config["training"]
    accumulation = int(training_config.get("gradient_accumulation_steps", 1))
    if accumulation < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1")
    max_epochs = int(training_config["max_epochs"])
    updates_per_epoch = math.ceil(len(train_records) / accumulation)
    total_steps = max(1, int(training_config.get("max_steps") or updates_per_epoch * max_epochs))
    trainable = [parameter for parameter in system.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=training_config["learning_rate"],
        betas=tuple(training_config["betas"]),
        weight_decay=training_config["weight_decay"],
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, training_config["warmup_steps"], total_steps
    )
    modules = _trainable_modules(system)
    fingerprint = config_fingerprint(config)
    state = TrainerState(config_fingerprint=fingerprint)
    if resume_from is not None:
        state = load_training_checkpoint(
            modules,
            _resume_path(resume_from),
            optimizer=optimizer,
            scheduler=scheduler,
            expected_fingerprints={"config": fingerprint},
            restore_rng=True,
        )
        if not state.exact_resume:
            raise ValueError("training resume requires a checkpoint v2 exact-resume state")

    patience = training_config.get("early_stopping_patience")
    if validation_records and patience is None:
        raise ValueError("PAPER_UNSPECIFIED early-stopping patience")
    early_stopping = None
    if validation_records:
        best = None if not math.isfinite(state.best_validation) else state.best_validation
        early_stopping = EarlyStopping(
            mode=training_config.get("early_stopping_mode", "min"),
            patience=int(patience),
            min_delta=float(training_config.get("early_stopping_min_delta", 0.0)),
            warmup_observations=int(
                training_config.get("early_stopping_warmup_validations", 0)
            ),
            observations=state.validation_observations,
            best=best,
            bad_validations=state.bad_validations,
            stopped=state.early_stopped,
        )

    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_log.jsonl"
    if resume_from is None:
        log_path.unlink(missing_ok=True)
    optimizer.zero_grad(set_to_none=True)
    micro_step = state.micro_step
    optimizer_step = state.optimizer_step
    start_epoch = state.epoch
    start_cursor = state.record_cursor
    seed = int(training_config.get("seed", config.get("seed", 42)))
    offset = int(training_config.get("initial_curriculum_step", 0))
    early_stopping_metric = training_config.get(
        "early_stopping_metric", "overall.losses.lm"
    )
    last_trace = None
    last_validation_report = None
    stopped = state.early_stopped
    window_traces: list[dict] = []

    def save_state(epoch: int, cursor: int, *, best: bool = False) -> Path:
        current = TrainerState(
            epoch=epoch,
            record_cursor=cursor,
            micro_step=micro_step,
            optimizer_step=optimizer_step,
            curriculum_step=offset + optimizer_step,
            sampler_state={"seed": seed, "epoch": epoch, "cursor": cursor},
            best_validation=(
                float("inf")
                if early_stopping is None or early_stopping.best is None
                else early_stopping.best
            ),
            bad_validations=0 if early_stopping is None else early_stopping.bad_validations,
            validation_observations=(
                0 if early_stopping is None else early_stopping.observations
            ),
            early_stopped=False if early_stopping is None else early_stopping.stopped,
            config_fingerprint=fingerprint,
            wandb_run_id=None if tracker is None else tracker.run_id,
            wandb_mode="disabled" if tracker is None else tracker.mode,
        )
        destination = checkpoint_dir / f"step-{optimizer_step:08d}"
        save_training_checkpoint(
            modules,
            destination,
            state=current,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        pointer = {"path": destination.name, "optimizer_step": optimizer_step}
        atomic_write_json(checkpoint_dir / "last.json", pointer)
        if best:
            atomic_write_json(checkpoint_dir / "best.json", pointer)
        if tracker is not None:
            tracker.log_checkpoint(destination)
        return destination

    for epoch in range(start_epoch, max_epochs):
        system.train()
        system.base_model.eval()
        order = _epoch_order(len(train_records), seed, epoch)
        cursor_begin = start_cursor if epoch == start_epoch else 0
        pending = 0
        for cursor in range(cursor_begin, len(order)):
            record = train_records[order[cursor]]
            bank = _memory(system, tokenizer, record, config)
            record_totals = []
            record_traces = []
            for qa in record.qa_pairs[: config["data"]["max_questions_per_sample"]]:
                curriculum_step = offset + optimizer_step
                bundle, _lm_value, trace = _one_question(
                    system, tokenizer, record, qa, bank, config, curriculum_step
                )
                record_totals.append(bundle.total)
                record_traces.append(trace)
                last_trace = trace
            if not record_totals:
                continue
            (torch.stack(record_totals).mean() / accumulation).backward()
            micro_step += 1
            pending += 1
            window_traces.extend(record_traces)
            at_epoch_end = cursor + 1 == len(order)
            if pending < accumulation and not at_epoch_end:
                continue

            if at_epoch_end and pending < accumulation:
                correction = accumulation / pending
                for parameter in trainable:
                    if parameter.grad is not None:
                        parameter.grad.mul_(correction)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable, training_config["gradient_clip"]
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            pending = 0
            optimizer_step += 1
            next_epoch = epoch + 1 if at_epoch_end else epoch
            next_cursor = 0 if at_epoch_end else cursor + 1
            validation_every = training_config.get("validation_every_steps")
            validation_due = bool(
                validation_records
                and ((validation_every and optimizer_step % validation_every == 0) or at_epoch_end)
            )
            improved = False
            if validation_due:
                last_validation_report = _validation_report(
                    system, tokenizer, validation_records, config, offset + optimizer_step
                )
                value = _metric_value(last_validation_report, early_stopping_metric)
                observation = early_stopping.observe(value)
                improved = observation.improved
                stopped = observation.should_stop
                if observation.nonfinite:
                    LOGGER.warning("non-finite early-stopping metric at optimizer step %s", optimizer_step)
                system.train()
                system.base_model.eval()

            event = {
                "epoch": epoch,
                "micro_step": micro_step,
                "optimizer_step": optimizer_step,
                "step": optimizer_step,
                "curriculum_step": offset + optimizer_step,
                "learning_rate": float(scheduler.get_last_lr()[0]),
                "gradient_norm": float(gradient_norm),
                "elapsed_seconds": time.perf_counter() - started,
                "cuda_allocated_bytes": (
                    torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                ),
                "cuda_reserved_bytes": (
                    torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
                ),
                "train": _split_trace_report(window_traces),
                "validation": last_validation_report if validation_due else None,
            }
            with log_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(event) + "\n")
            if tracker is not None:
                tracker.log_metrics(
                    {
                        "train/total_loss": event["train"]["overall"].get("total_loss"),
                        "train/gradient_norm": event["gradient_norm"],
                        "train/learning_rate": event["learning_rate"],
                    },
                    step=optimizer_step,
                )
            window_traces = []
            max_steps_reached = optimizer_step >= total_steps
            requested_stop = (
                stop_after_optimizer_steps is not None
                and optimizer_step >= stop_after_optimizer_steps
            )
            checkpoint_every = int(training_config.get("checkpoint_every_steps", 1000))
            if (
                improved
                or validation_due
                or at_epoch_end
                or max_steps_reached
                or requested_stop
                or optimizer_step % checkpoint_every == 0
            ):
                save_state(next_epoch, next_cursor, best=improved)
            if stopped or max_steps_reached or requested_stop:
                break
        start_cursor = 0
        if stopped or optimizer_step >= total_steps or (
            stop_after_optimizer_steps is not None
            and optimizer_step >= stop_after_optimizer_steps
        ):
            break

    if optimizer_step and not (checkpoint_dir / "last.json").exists():
        save_state(min(max_epochs, start_epoch + 1), 0)
    best_validation = (
        float("inf")
        if early_stopping is None or early_stopping.best is None
        else early_stopping.best
    )
    return {
        "steps": optimizer_step,
        "micro_steps": micro_step,
        "optimizer_steps": optimizer_step,
        "curriculum_step": offset + optimizer_step,
        "early_stopped": stopped,
        "early_stopping_metric": early_stopping_metric,
        "best_validation_metric": best_validation,
        "best_validation_lm": best_validation,
        "last_trace": last_trace,
        "last_validation": last_validation_report,
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
def _validation_report(system, tokenizer, records, config, step):
    if not records:
        return _split_trace_report([])
    system.eval()
    traces = []
    for record in records:
        bank = _memory(system, tokenizer, record, config)
        for qa in record.qa_pairs[: config["data"]["max_questions_per_sample"]]:
            _bundle, _lm, trace = _one_question(
                system, tokenizer, record, qa, bank, config, step
            )
            traces.append(trace)
    return _split_trace_report(traces)

from __future__ import annotations

import argparse
import copy
import json
import re
import unicodedata
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from doc_to_atom.compiler.memory import MemoryBank
from doc_to_atom.composer.model import ComposedMemory
from doc_to_atom.config import load_config, load_project_env
from doc_to_atom.data.io import read_jsonl, validate_jsonl
from doc_to_atom.data.targets import resolve_target
from doc_to_atom.evaluation.metrics import refusal_f1, rouge_l, word_f1
from doc_to_atom.training.factory import build_system
from doc_to_atom.training.loop import _answer_logits, _causal_example, _tokenize


def _answer_text(answer: str | list[str] | None) -> str:
    if isinstance(answer, list):
        return str(answer[0]) if answer else ""
    return str(answer or "")


def _normalized(text: str) -> str:
    return " ".join(
        re.findall(r"\w+", unicodedata.normalize("NFKC", text).lower(), flags=re.UNICODE)
    )


def _grad_norm(parameters) -> float:
    squares = [
        parameter.grad.float().pow(2).sum()
        for parameter in parameters
        if parameter.grad is not None
    ]
    return float(torch.stack(squares).sum().sqrt()) if squares else 0.0


def _factor_stats(composed: ComposedMemory, include_grad: bool = False) -> dict:
    result = {}
    for key in composed.factors_a:
        a = composed.factors_a[key]
        b = composed.factors_b[key]
        delta = b @ a
        result[key] = {
            "a_norm": float(a.detach().float().norm()),
            "b_norm": float(b.detach().float().norm()),
            "delta_w_norm_unscaled": float(delta.detach().float().norm()),
            "a_mean": float(a.detach().float().mean()),
            "a_std": float(a.detach().float().std()),
            "b_mean": float(b.detach().float().mean()),
            "b_std": float(b.detach().float().std()),
            "grad_a_norm": (
                float(a.grad.detach().float().norm())
                if include_grad and a.grad is not None
                else None
            ),
            "grad_b_norm": (
                float(b.grad.detach().float().norm())
                if include_grad and b.grad is not None
                else None
            ),
        }
    return result


class DirectLoRA(nn.Module):
    def __init__(self, targets, rank: int, a_std: float, device: torch.device):
        super().__init__()
        self.targets = targets
        self.a = nn.ParameterDict()
        self.b = nn.ParameterDict()
        for target in targets:
            safe = target.key.replace(":", "__")
            self.a[safe] = nn.Parameter(
                torch.empty(rank, target.input_dim, device=device, dtype=torch.float32)
            )
            self.b[safe] = nn.Parameter(
                torch.zeros(target.output_dim, rank, device=device, dtype=torch.float32)
            )
            nn.init.normal_(self.a[safe], mean=0.0, std=a_std)

    def composed(self) -> ComposedMemory:
        factors_a = {}
        factors_b = {}
        gates = {}
        for target in self.targets:
            safe = target.key.replace(":", "__")
            factors_a[target.key] = self.a[safe]
            factors_b[target.key] = self.b[safe]
            gates[target.key] = torch.ones((), device=self.a[safe].device)
        return ComposedMemory(factors_a, factors_b, gates, None)


def _prepared(tokenizer, prompt: str, answer: str, max_length: int, device):
    inputs, labels, count = _causal_example(tokenizer, prompt, answer, max_length, device)
    return inputs, labels[labels != -100][-count:], count


def _loss(system, prepared, teacher_answer, composed, temperature: float):
    inputs, answer_ids, answer_count = prepared
    with system.injector.activate(composed):
        logits = system.base_model(**inputs, use_cache=False).logits
        student = _answer_logits(logits, answer_count).float()
        common = min(student.shape[1], teacher_answer.shape[1])
        student = student[:, -common:]
        teacher = teacher_answer[:, -common:]
        ids = answer_ids[-common:]
        lm = F.cross_entropy(student.reshape(-1, student.size(-1)), ids)
        kl = F.kl_div(
            F.log_softmax(student / temperature, -1),
            F.softmax(teacher / temperature, -1),
            reduction="batchmean",
        )
        total = lm + kl
        total.backward()
    return float(lm.detach()), float(kl.detach())


@torch.inference_mode()
def _evaluate(
    system,
    tokenizer,
    prompt: str,
    answer: str,
    teacher_answer,
    composed: ComposedMemory | None,
    config,
):
    device = next(system.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    if composed is None:
        output = system.base_model.generate(
            **inputs, do_sample=False, max_new_tokens=config["evaluation"]["max_new_tokens"]
        )
    else:
        with system.injector.activate(composed):
            output = system.base_model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=config["evaluation"]["max_new_tokens"],
            )
    prediction = tokenizer.decode(
        output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()
    prepared = _prepared(tokenizer, prompt, answer, config["data"]["max_context_tokens"], device)
    model_inputs, answer_ids, answer_count = prepared
    if composed is None:
        logits = system.base_model(**model_inputs, use_cache=False).logits
    else:
        with system.injector.activate(composed):
            logits = system.base_model(**model_inputs, use_cache=False).logits
    student = _answer_logits(logits, answer_count).float()
    common = min(student.shape[1], teacher_answer.shape[1])
    student = student[:, -common:]
    teacher = teacher_answer[:, -common:]
    nll = F.cross_entropy(student.reshape(-1, student.size(-1)), answer_ids[-common:])
    kl = F.kl_div(
        F.log_softmax(student / config["training"]["distillation_temperature"], -1),
        F.softmax(teacher / config["training"]["distillation_temperature"], -1),
        reduction="batchmean",
    )
    exact = float(_normalized(prediction) == _normalized(answer))
    first_line_exact = float(
        _normalized(prediction.splitlines()[0] if prediction else "") == _normalized(answer)
    )
    answer_containment = float(
        bool(_normalized(answer)) and _normalized(answer) in _normalized(prediction)
    )
    return {
        "prediction": prediction,
        "word_f1": word_f1(prediction, answer),
        "rouge_l": rouge_l(prediction, answer),
        "exact": exact,
        "first_line_exact": first_line_exact,
        "answer_containment": answer_containment,
        "gold_answer_nll": float(nll),
        "teacher_student_kl": float(kl),
        "adapter_norm": None if composed is None else float(composed.adapter_norm()),
    }


def _teacher_answer(system, prepared):
    inputs, _answer_ids, answer_count = prepared
    if system.injector._active.get() is not None:
        raise AssertionError("teacher unexpectedly has an active adapter")
    with torch.no_grad():
        logits = system.base_model(**inputs, use_cache=False).logits
    return _answer_logits(logits, answer_count).float().detach()


def _compile(system, atom_inputs, atoms) -> MemoryBank:
    embeddings = system.encoder.encode_atoms(**atom_inputs)
    compiled = system.compiler(embeddings)
    return MemoryBank(**compiled.__dict__, atoms=atoms)


def _retain_factor_grads(composed: ComposedMemory) -> None:
    for tensor in [*composed.factors_a.values(), *composed.factors_b.values()]:
        tensor.retain_grad()


def _compiler_eval(system, tokenizer, atom_inputs, record, gold_index, answer, teacher, config):
    with torch.no_grad():
        bank = _compile(system, atom_inputs, record.atoms)
        index = torch.tensor([gold_index], device=bank.keys.device)
        weight = torch.ones(1, device=bank.keys.device, dtype=bank.keys.dtype)
        composed = system.composer(bank, index, weight)
    prompt = f"Question: {record.qa_pairs[0].question}\nAnswer:"
    return _evaluate(system, tokenizer, prompt, answer, teacher, composed, config), composed


def _irrelevant_eval(system, tokenizer, atom_inputs, record, qa, config):
    with torch.no_grad():
        bank = _compile(system, atom_inputs, record.atoms)
        indices = torch.arange(len(record.atoms), device=bank.keys.device)
        weights = torch.full((len(record.atoms),), 1.0 / len(record.atoms), device=bank.keys.device)
        composed = system.composer(bank, indices, weights)
    prompt = f"Question: {qa.question}\nAnswer:"
    result = _evaluate(
        system,
        tokenizer,
        prompt,
        "unanswerable",
        _teacher_answer(
            system,
            _prepared(
                tokenizer,
                f"Document:\n{record.context}\n\nQuestion: {qa.question}\nAnswer:",
                "unanswerable",
                config["data"]["max_context_tokens"],
                next(system.parameters()).device,
            ),
        ),
        composed,
        config,
    )
    result["refusal_f1"] = refusal_f1(result["prediction"])
    result["contains_unanswerable"] = "unanswerable" in result["prediction"].lower()
    return result


def _set_compiler_trainable(system) -> list[nn.Parameter]:
    for parameter in system.parameters():
        parameter.requires_grad_(False)
    for module in (system.encoder.projection, system.compiler):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    return [parameter for parameter in system.parameters() if parameter.requires_grad]


def _train_compiler(
    system,
    tokenizer,
    record,
    answer_qa,
    irrelevant_qa,
    atom_inputs,
    gold_index,
    answer_prepared,
    answer_teacher,
    irrelevant_prepared,
    irrelevant_teacher,
    config,
    steps: int,
    irrelevant_weight: float,
    learning_rate: float,
    log_steps: set[int],
):
    trainable = _set_compiler_trainable(system)
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=0.0)
    temperature = config["training"]["distillation_temperature"]
    trace = []
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        bank = _compile(system, atom_inputs, record.atoms)
        index = torch.tensor([gold_index], device=bank.keys.device)
        one = torch.ones(1, device=bank.keys.device, dtype=bank.keys.dtype)
        forced = system.composer(bank, index, one)
        _retain_factor_grads(forced)
        lm, kl = _loss(system, answer_prepared, answer_teacher, forced, temperature)
        irrel_lm = irrel_kl = irrel_suppression = 0.0
        if irrelevant_weight:
            # A separate compiler graph lets the answer backward complete while the
            # irrelevant gradient is accumulated before the shared optimizer step.
            irrelevant_bank = _compile(system, atom_inputs, record.atoms)
            all_indices = torch.arange(len(record.atoms), device=irrelevant_bank.keys.device)
            all_weights = torch.full(
                (len(record.atoms),),
                1.0 / len(record.atoms),
                device=irrelevant_bank.keys.device,
                dtype=irrelevant_bank.keys.dtype,
            )
            irrelevant_memory = system.composer(irrelevant_bank, all_indices, all_weights)
            with system.injector.activate(irrelevant_memory):
                inputs, answer_ids, answer_count = irrelevant_prepared
                logits = system.base_model(**inputs, use_cache=False).logits
                student = _answer_logits(logits, answer_count).float()
                common = min(student.shape[1], irrelevant_teacher.shape[1])
                student = student[:, -common:]
                teacher = irrelevant_teacher[:, -common:]
                loss_lm = F.cross_entropy(
                    student.reshape(-1, student.size(-1)), answer_ids[-common:]
                )
                loss_kl = F.kl_div(
                    F.log_softmax(student / temperature, -1),
                    F.softmax(teacher / temperature, -1),
                    reduction="batchmean",
                )
                suppression = sum(
                    value.float().pow(2).sum()
                    for value in [
                        *irrelevant_memory.factors_a.values(),
                        *irrelevant_memory.factors_b.values(),
                    ]
                )
                irrel_total = irrelevant_weight * (
                    loss_lm
                    + loss_kl
                    + config["training"]["loss_weights"]["irrelevant"] * suppression
                )
                irrel_total.backward()
            irrel_lm = float(loss_lm.detach())
            irrel_kl = float(loss_kl.detach())
            irrel_suppression = float(suppression.detach())
        grad_norm = _grad_norm(trainable)
        head_grad_norm = _grad_norm(system.compiler.parameters())
        factor_stats = _factor_stats(forced, include_grad=True)
        optimizer.step()
        if step in log_steps or step == steps:
            evaluation, current = _compiler_eval(
                system,
                tokenizer,
                atom_inputs,
                record,
                gold_index,
                resolve_target(answer_qa).text,
                answer_teacher,
                config,
            )
            trace.append(
                {
                    "step": step,
                    "lm": lm,
                    "distillation": kl,
                    "irrelevant_lm": irrel_lm,
                    "irrelevant_distillation": irrel_kl,
                    "irrelevant_suppression": irrel_suppression,
                    "gradient_norm": grad_norm,
                    "compiler_gradient_norm": head_grad_norm,
                    "evaluation": evaluation,
                    "factor_stats_before_update": factor_stats,
                    "factor_stats_after_update": _factor_stats(current),
                }
            )
    return trace


def main() -> None:
    load_project_env()
    parser = argparse.ArgumentParser(description="Parametric-memory transfer isolation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--record-id", default="squad_train_7")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--direct-lr", type=float, default=0.01)
    parser.add_argument("--compiler-lr", type=float, default=0.0001)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "results.json").exists():
        raise FileExistsError(f"refusing to overwrite completed run: {output}")
    stats = validate_jsonl(args.data, reject_span_mismatch=True)
    records = read_jsonl(args.data)
    record = next(item for item in records if item.record_id == args.record_id)
    if record.split != "train":
        raise ValueError("diagnostic record must belong to train split")
    answer_qa = next(qa for qa in record.qa_pairs if not qa.is_irrelevant)
    irrelevant_qa = next(qa for qa in record.qa_pairs if qa.is_generated_irrelevant)
    answer = resolve_target(answer_qa).text
    if len(answer_qa.gold_atom_ids) != 1:
        raise ValueError("this controlled run requires exactly one gold atom")

    torch.manual_seed(20261011)
    config = load_config(args.config)
    system, tokenizer, targets = build_system(config, enable_reranker=False)
    system.eval()
    device = next(system.parameters()).device
    initial_projection = copy.deepcopy(system.encoder.projection.state_dict())
    initial_compiler = copy.deepcopy(system.compiler.state_dict())
    base_frozen = all(not parameter.requires_grad for parameter in system.base_model.parameters())
    gold_index = next(
        index
        for index, atom in enumerate(record.atoms)
        if atom.atom_id == answer_qa.gold_atom_ids[0]
    )
    gold_text = record.atoms[gold_index].content
    query_prompt = f"Question: {answer_qa.question}\nAnswer:"
    teacher_prompt = f"Document:\n{record.context}\n\nQuestion: {answer_qa.question}\nAnswer:"
    gold_text_prompt = f"Gold atom:\n{gold_text}\n\nQuestion: {answer_qa.question}\nAnswer:"
    answer_prepared = _prepared(
        tokenizer, query_prompt, answer, config["data"]["max_context_tokens"], device
    )
    teacher_prepared = _prepared(
        tokenizer, teacher_prompt, answer, config["data"]["max_context_tokens"], device
    )
    answer_teacher = _teacher_answer(system, teacher_prepared)
    irrelevant_prompt = f"Question: {irrelevant_qa.question}\nAnswer:"
    irrelevant_teacher_prompt = (
        f"Document:\n{record.context}\n\nQuestion: {irrelevant_qa.question}\nAnswer:"
    )
    irrelevant_prepared = _prepared(
        tokenizer,
        irrelevant_prompt,
        "unanswerable",
        config["data"]["max_context_tokens"],
        device,
    )
    irrelevant_teacher = _teacher_answer(
        system,
        _prepared(
            tokenizer,
            irrelevant_teacher_prompt,
            "unanswerable",
            config["data"]["max_context_tokens"],
            device,
        ),
    )
    atom_inputs = _tokenize(
        tokenizer,
        [atom.content for atom in record.atoms],
        config["data"]["max_atom_tokens"],
        device,
    )
    baseline = {
        "base_query_only": _evaluate(
            system, tokenizer, query_prompt, answer, answer_teacher, None, config
        ),
        "base_gold_atom_text": _evaluate(
            system, tokenizer, gold_text_prompt, answer, answer_teacher, None, config
        ),
        "full_context_teacher": _evaluate(
            system, tokenizer, teacher_prompt, answer, answer_teacher, None, config
        ),
    }

    log_steps = {1, 2, 5, 10, 20, args.steps}
    for parameter in system.parameters():
        parameter.requires_grad_(False)
    direct = DirectLoRA(
        targets,
        config["compiler"]["lora_rank"],
        config["compiler"]["lora_a_std"],
        device,
    )
    direct_optimizer = torch.optim.AdamW(direct.parameters(), lr=args.direct_lr, weight_decay=0.0)
    direct_trace = []
    for step in range(1, args.steps + 1):
        direct_optimizer.zero_grad(set_to_none=True)
        composed = direct.composed()
        lm, kl = _loss(
            system,
            answer_prepared,
            answer_teacher,
            composed,
            config["training"]["distillation_temperature"],
        )
        grad_norm = _grad_norm(direct.parameters())
        factor_stats = _factor_stats(composed, include_grad=True)
        direct_optimizer.step()
        if step in log_steps:
            current = direct.composed()
            direct_trace.append(
                {
                    "step": step,
                    "lm": lm,
                    "distillation": kl,
                    "gradient_norm": grad_norm,
                    "evaluation": _evaluate(
                        system,
                        tokenizer,
                        query_prompt,
                        answer,
                        answer_teacher,
                        current,
                        config,
                    ),
                    "factor_stats_before_update": factor_stats,
                    "factor_stats_after_update": _factor_stats(current),
                }
            )
    direct_final = direct_trace[-1]["evaluation"]
    direct_final_stats = direct_trace[-1]["factor_stats_after_update"]
    direct_grad_nonzero = any(
        item["grad_a_norm"] and item["grad_b_norm"]
        for item in direct_trace[-1]["factor_stats_before_update"].values()
    )
    del direct_optimizer, direct
    torch.cuda.empty_cache()

    def reset_compiler() -> None:
        system.encoder.projection.load_state_dict(initial_projection)
        system.compiler.load_state_dict(initial_compiler)

    reset_compiler()
    answerable_trace = _train_compiler(
        system,
        tokenizer,
        record,
        answer_qa,
        irrelevant_qa,
        atom_inputs,
        gold_index,
        answer_prepared,
        answer_teacher,
        irrelevant_prepared,
        irrelevant_teacher,
        config,
        args.steps,
        0.0,
        args.compiler_lr,
        log_steps,
    )
    answerable_final = answerable_trace[-1]["evaluation"]
    answerable_irrelevant = _irrelevant_eval(
        system, tokenizer, atom_inputs, record, irrelevant_qa, config
    )
    answerable_state = {
        "projection": copy.deepcopy(system.encoder.projection.state_dict()),
        "compiler": copy.deepcopy(system.compiler.state_dict()),
    }

    reset_compiler()
    mixed_trace = _train_compiler(
        system,
        tokenizer,
        record,
        answer_qa,
        irrelevant_qa,
        atom_inputs,
        gold_index,
        answer_prepared,
        answer_teacher,
        irrelevant_prepared,
        irrelevant_teacher,
        config,
        args.steps,
        1.0,
        args.compiler_lr,
        log_steps,
    )
    mixed_final = mixed_trace[-1]["evaluation"]
    mixed_irrelevant = _irrelevant_eval(
        system, tokenizer, atom_inputs, record, irrelevant_qa, config
    )

    refusal_reintroduction = []
    if answerable_final["answer_containment"]:
        system.encoder.projection.load_state_dict(answerable_state["projection"])
        system.compiler.load_state_dict(answerable_state["compiler"])
        for weight in (0.25, 0.5, 1.0):
            phase_trace = _train_compiler(
                system,
                tokenizer,
                record,
                answer_qa,
                irrelevant_qa,
                atom_inputs,
                gold_index,
                answer_prepared,
                answer_teacher,
                irrelevant_prepared,
                irrelevant_teacher,
                config,
                10,
                weight,
                args.compiler_lr,
                {10},
            )
            refusal_reintroduction.append(
                {
                    "irrelevant_weight": weight,
                    "answer": phase_trace[-1]["evaluation"],
                    "irrelevant": _irrelevant_eval(
                        system, tokenizer, atom_inputs, record, irrelevant_qa, config
                    ),
                }
            )

    result = {
        "data": stats,
        "record_id": record.record_id,
        "split": record.split,
        "question": answer_qa.question,
        "answer": answer,
        "gold_atom_id": answer_qa.gold_atom_ids[0],
        "gold_atom_text": gold_text,
        "irrelevant_question": irrelevant_qa.question,
        "steps_per_main_mode": args.steps,
        "learning_rates": {"direct": args.direct_lr, "compiler": args.compiler_lr},
        "objective": "answer LM + teacher KL; mixed adds weighted irrelevant LM + KL + 0.1 suppression",
        "irrelevant_memory_strategy": "uniform composition over all document atoms; router bypassed",
        "baseline": baseline,
        "direct_lora": {"final": direct_final, "trace": direct_trace},
        "compiler_answerable_only": {
            "final": answerable_final,
            "irrelevant_evaluation": answerable_irrelevant,
            "trace": answerable_trace,
        },
        "compiler_mixed": {
            "final": mixed_final,
            "irrelevant_evaluation": mixed_irrelevant,
            "trace": mixed_trace,
        },
        "refusal_reintroduction": refusal_reintroduction,
        "technical_checks": {
            "base_model_frozen": base_frozen,
            "base_parameter_count_with_grad": sum(
                parameter.requires_grad for parameter in system.base_model.parameters()
            ),
            "target_count": len(targets),
            "target_keys": [target.key for target in targets],
            "rank": config["compiler"]["lora_rank"],
            "alpha": config["compiler"]["lora_alpha"],
            "injection_scale_alpha_over_rank": system.injector.scale,
            "registered_dynamic_hooks": len(system.injector._handles),
            "teacher_adapter_inactive": system.injector._active.get() is None,
            "direct_gradients_nonzero_a_and_b": direct_grad_nonzero,
            "compiler_gradient_norm_final": answerable_trace[-1]["compiler_gradient_norm"],
            "compiler_outputs_require_grad": all(
                value["grad_b_norm"] is not None
                for value in answerable_trace[-1]["factor_stats_before_update"].values()
            ),
            "direct_final_factor_stats": direct_final_stats,
            "compiler_final_factor_stats": answerable_trace[-1]["factor_stats_after_update"],
        },
    }
    (output / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "record_id": record.record_id,
                "baseline": baseline,
                "direct_final": direct_final,
                "compiler_answerable_only_final": answerable_final,
                "compiler_mixed_final": mixed_final,
                "refusal_reintroduction": refusal_reintroduction,
                "technical_checks": {
                    key: value
                    for key, value in result["technical_checks"].items()
                    if not key.endswith("factor_stats")
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

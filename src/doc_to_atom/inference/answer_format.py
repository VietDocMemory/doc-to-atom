"""Versioned, gold-independent short-answer generation policy."""

from __future__ import annotations

import torch
from transformers import StoppingCriteria, StoppingCriteriaList


GENERATION_PROTOCOL = "short_answer_v1"


def build_qa_prompt(question: str, *, context: str | None = None) -> str:
    instruction = (
        "Answer with one short phrase only. Do not explain. "
        "If the available information cannot answer the question, write unanswerable."
    )
    document = f"Document:\n{context}\n\n" if context is not None else ""
    return f"{instruction}\n{document}Question: {question}\nAnswer:"


def _completed_answer_line(text: str) -> bool:
    return any(
        line.strip() and line.endswith(("\n", "\r"))
        for line in text.splitlines(keepends=True)
    )


def _first_answer_line(text: str) -> str:
    return next((line.strip() for line in text.splitlines() if line.strip()), "")


class _StopAfterAnswerLine(StoppingCriteria):
    def __init__(self, tokenizer, prompt_length: int) -> None:
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
        continuation = input_ids[0, self.prompt_length:]
        text = self.tokenizer.decode(continuation, skip_special_tokens=True)
        return _completed_answer_line(text)


@torch.no_grad()
def generate_short_answer(model, tokenizer, prompt: str, *, max_new_tokens: int) -> dict:
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    device = getattr(model, "device", None)
    if device is None:
        device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_length = inputs["input_ids"].shape[1]
    stop = _StopAfterAnswerLine(tokenizer, prompt_length)
    output = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        stopping_criteria=StoppingCriteriaList([stop]),
    )
    generated_ids = output[0, prompt_length:]
    raw_generation = tokenizer.decode(generated_ids, skip_special_tokens=True)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    stop_reason = (
        "answer_line" if _completed_answer_line(raw_generation) else
        "eos" if eos_id is not None and eos_id in generated_ids.tolist() else
        "max_new_tokens" if len(generated_ids) >= max_new_tokens else
        "model_stop"
    )
    return {
        "prediction": _first_answer_line(raw_generation),
        "raw_generation": raw_generation,
        "stop_reason": stop_reason,
        "generation_protocol": GENERATION_PROTOCOL,
    }

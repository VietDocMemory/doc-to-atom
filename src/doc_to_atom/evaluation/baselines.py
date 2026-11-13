from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from doc_to_atom.inference.answer_format import build_qa_prompt, generate_short_answer


class Baseline(Protocol):
    name: str

    def answer(self, document: str, question: str) -> str: ...


@dataclass
class BaseQueryOnly:
    model: object
    tokenizer: object
    max_new_tokens: int = 128
    name: str = "base_query_only"

    def answer(self, document: str, question: str) -> str:
        prompt = build_qa_prompt(question)
        return generate_short_answer(
            self.model, self.tokenizer, prompt, max_new_tokens=self.max_new_tokens
        )["prediction"]


@dataclass
class BaseWithContext:
    model: object
    tokenizer: object
    max_new_tokens: int = 128
    name: str = "base_context"

    def answer(self, document: str, question: str) -> str:
        prompt = build_qa_prompt(question, context=document)
        return generate_short_answer(
            self.model, self.tokenizer, prompt, max_new_tokens=self.max_new_tokens
        )["prediction"]


@dataclass
class DocToLoRABaseline:
    """Adapter for official, raw-trained, or atomized-trained Doc-to-LoRA checkpoints."""

    d2l_model: object
    tokenizer: object
    variant: str
    max_new_tokens: int = 128
    prompt_protocol: str = "raw_v1"

    @property
    def name(self) -> str:
        if self.variant not in {"official", "raw", "atomized"}:
            raise ValueError("D2L variant must be official, raw, or atomized")
        return f"d2l_{self.variant}"

    def answer(self, document: str, question: str) -> str:
        if self.prompt_protocol not in {"raw_v1", "chat_v1"}:
            raise ValueError("D2L prompt_protocol must be raw_v1 or chat_v1")
        if self.prompt_protocol == "chat_v1":
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=False, add_generation_prompt=True,
            )
        else:
            prompt = question
        self.d2l_model.reset()
        try:
            self.d2l_model.internalize(document)
            return generate_short_answer(
                self.d2l_model, self.tokenizer, prompt, max_new_tokens=self.max_new_tokens,
                add_special_tokens=False if self.prompt_protocol == "chat_v1" else None,
            )["prediction"]
        finally:
            self.d2l_model.reset()


@dataclass
class Doc2AtomBaseline:
    engine: object
    bank: object
    name: str = "doc2atom"

    def answer(self, document: str, question: str) -> str:
        return self.engine.answer(question, self.bank)["prediction"]

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

    @property
    def name(self) -> str:
        if self.variant not in {"official", "raw", "atomized"}:
            raise ValueError("D2L variant must be official, raw, or atomized")
        return f"d2l_{self.variant}"

    def answer(self, document: str, question: str) -> str:
        self.d2l_model.reset()
        self.d2l_model.internalize(document)
        return generate_short_answer(
            self.d2l_model, self.tokenizer, question, max_new_tokens=self.max_new_tokens
        )["prediction"]


@dataclass
class Doc2AtomBaseline:
    engine: object
    bank: object
    name: str = "doc2atom"

    def answer(self, document: str, question: str) -> str:
        return self.engine.answer(question, self.bank)["prediction"]

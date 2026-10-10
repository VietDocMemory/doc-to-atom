from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class Baseline(Protocol):
    name: str

    def answer(self, document: str, question: str) -> str: ...


@dataclass
class BaseWithContext:
    model: object
    tokenizer: object
    max_new_tokens: int = 128
    name: str = "base_context"

    def answer(self, document: str, question: str) -> str:
        prompt = f"Document:\n{document}\n\nQuestion: {question}\nAnswer:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(
            next(self.model.parameters()).device
        )
        output = self.model.generate(**inputs, do_sample=False, max_new_tokens=self.max_new_tokens)
        return self.tokenizer.decode(
            output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )


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
        inputs = self.tokenizer(question, return_tensors="pt").to(self.d2l_model.device)
        output = self.d2l_model.generate(
            **inputs, do_sample=False, max_new_tokens=self.max_new_tokens
        )
        return self.tokenizer.decode(
            output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )


@dataclass
class Doc2AtomBaseline:
    engine: object
    bank: object
    name: str = "doc2atom"

    def answer(self, document: str, question: str) -> str:
        return self.engine.answer(question, self.bank)["prediction"]

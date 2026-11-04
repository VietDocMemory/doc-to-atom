"""Versioned training behavior: never silently change an old checkpoint's objective."""
from __future__ import annotations

import torch

from doc_to_atom.inference.answer_format import build_qa_prompt

LEGACY_PROTOCOL = "legacy_v1"
ALIGNED_PROTOCOL = "aligned_qa_v2"
PROTOCOLS = {LEGACY_PROTOCOL, ALIGNED_PROTOCOL}
QUESTION_REDUCTIONS = {"question_mean", "group_balanced"}


def objective_protocol(config: dict) -> str:
    value = config.get("training", {}).get("objective_protocol", LEGACY_PROTOCOL)
    if value not in PROTOCOLS:
        raise ValueError(f"unknown training objective_protocol: {value}")
    return value


def training_texts(question: str, context: str, answer: str, config: dict) -> tuple[str, str, str]:
    if objective_protocol(config) == LEGACY_PROTOCOL:
        return (f"Document:\n{context}\n\nQuestion: {question}\nAnswer:",
                f"Question: {question}\nAnswer:", answer)
    # Same gold-independent instructions as generation. Termination belongs to
    # the LM target, not the immutable source answer stored in annotation data.
    return (build_qa_prompt(question, context=context), build_qa_prompt(question),
            answer.rstrip("\r\n") + "\n")


def reduce_question_losses(losses: list, irrelevant: list[bool], reduction: str = "question_mean"):
    if not losses or len(losses) != len(irrelevant):
        raise ValueError("Question losses and group labels must be nonempty and aligned")
    if reduction not in QUESTION_REDUCTIONS:
        raise ValueError(f"Unknown question loss reduction: {reduction}")
    if reduction == "question_mean":
        return torch.stack(losses).mean()
    # Equal total influence for available answerable/refusal groups, regardless
    # of how many auxiliary probes were generated for this document.
    groups = [[value for value, flag in zip(losses, irrelevant, strict=True) if flag == group]
              for group in (False, True)]
    return torch.stack([torch.stack(group).mean() for group in groups if group]).mean()

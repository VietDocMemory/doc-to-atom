from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from doc_to_atom.atomization.models import Atom
from doc_to_atom.compiler.memory import MemoryBank
from doc_to_atom.compiler.model import AtomMemoryCompiler
from doc_to_atom.composer.model import ComposedMemory, MemoryComposer
from doc_to_atom.encoder.shared import SharedAtomQueryEncoder
from doc_to_atom.injection.lora import DynamicLoRAInjector, freeze_base_model
from doc_to_atom.router.model import RouterOutput, TwoStageRouter


@dataclass
class StudentOutput:
    logits: torch.Tensor
    router: RouterOutput
    composed: ComposedMemory


class TeacherStudentSystem(nn.Module):
    """Same frozen base model: teacher gets context+query, student gets query+memory."""

    def __init__(
        self,
        base_model: nn.Module,
        encoder: SharedAtomQueryEncoder,
        compiler: AtomMemoryCompiler,
        router: TwoStageRouter,
        composer: MemoryComposer,
        injector: DynamicLoRAInjector,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.encoder = encoder
        self.compiler = compiler
        self.router = router
        self.composer = composer
        self.injector = injector
        freeze_base_model(self.base_model)

    def forward(self, operation: str, **kwargs):
        """Dispatch all differentiable paths through the distributed wrapper.

        DDP and DeepSpeed only install their synchronization hooks around
        ``Module.__call__``. Keeping these operations behind one forward entry
        point prevents trainable submodules from silently bypassing those hooks.
        """
        operations = {
            "build_memory": self.build_memory,
            "teacher": self.teacher_forward,
            "student": self.student_forward,
            "forced_gold": self.forced_gold_forward,
        }
        try:
            implementation = operations[operation]
        except KeyError as error:
            raise ValueError(
                f"unknown TeacherStudentSystem operation: {operation!r}"
            ) from error
        return implementation(**kwargs)

    def build_memory(self, atom_inputs: dict[str, torch.Tensor], atoms: list[Atom]) -> MemoryBank:
        embeddings = self.encoder.encode_atoms(**atom_inputs)
        compiled = self.compiler(embeddings)
        return MemoryBank(**compiled.__dict__, atoms=atoms)

    def teacher_forward(self, full_context_query_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            return self.base_model(**full_context_query_inputs, use_cache=False).logits

    def student_forward(
        self,
        query_inputs: dict[str, torch.Tensor],
        query_embedding_inputs: dict[str, torch.Tensor],
        query_text: str,
        bank: MemoryBank,
        top_k: int,
    ) -> StudentOutput:
        query_embedding = self.encoder.encode_queries(**query_embedding_inputs)
        routed = self.router(query_embedding, bank.keys, bank.atoms or [], query_text, top_k)
        composed = self.composer(bank, routed.selected_indices, routed.routing_weights)
        with self.injector.activate(composed):
            logits = self.base_model(**query_inputs, use_cache=False).logits
        return StudentOutput(logits=logits, router=routed, composed=composed)

    def forced_gold_forward(
        self,
        query_inputs: dict[str, torch.Tensor],
        bank: MemoryBank,
        selected_indices: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        composed = self.composer(bank, selected_indices, routing_weights)
        with self.injector.activate(composed):
            return self.base_model(**query_inputs, use_cache=False).logits

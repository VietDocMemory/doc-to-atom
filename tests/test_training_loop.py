from pathlib import Path

import torch
from conftest import ToyLM

from doc_to_atom.atomization.models import Atom, QuestionAnnotation, StructuredRecord
from doc_to_atom.compiler.model import AtomMemoryCompiler
from doc_to_atom.composer.model import MemoryComposer
from doc_to_atom.config import load_config
from doc_to_atom.encoder.shared import SharedAtomQueryEncoder
from doc_to_atom.injection.lora import DynamicLoRAInjector, discover_target_shapes
from doc_to_atom.router.model import TwoStageRouter
from doc_to_atom.training.loop import run_training
from doc_to_atom.training.system import TeacherStudentSystem


class ToyTokenizer:
    def _ids(self, text):
        return [1] + [2 + ord(char) % 25 for char in text]

    def __call__(
        self,
        texts,
        padding=False,
        truncation=False,
        max_length=None,
        return_tensors=None,
        add_special_tokens=True,
    ):
        if isinstance(texts, str):
            ids = self._ids(texts)
            if truncation and max_length:
                ids = ids[:max_length]
            if return_tensors == "pt":
                tensor = torch.tensor([ids])
                return {"input_ids": tensor, "attention_mask": torch.ones_like(tensor)}
            return {"input_ids": ids}
        sequences = [
            self._ids(text)[:max_length] if max_length else self._ids(text) for text in texts
        ]
        width = max(map(len, sequences))
        ids = torch.zeros(len(sequences), width, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for index, sequence in enumerate(sequences):
            ids[index, : len(sequence)] = torch.tensor(sequence)
            mask[index, : len(sequence)] = 1
        return {"input_ids": ids, "attention_mask": mask}


def _record(split):
    atoms = [
        Atom(
            atom_id="a0",
            semantic_type="fact_claim",
            content="Alpha",
            source_span="Alpha",
            retrieval_text="Alpha",
            is_answer_bearing=True,
            abstraction_level="evidence",
            confidence=0.9,
        )
    ]
    qa = QuestionAnnotation(
        question_id="q0",
        question="What?",
        answer="Alpha",
        question_type="single_hop",
        gold_atom_ids=["a0"],
        relevant_atom_ids=["a0"],
    )
    return StructuredRecord(
        record_id=f"r-{split}", split=split, context="Alpha", atoms=atoms, qa_pairs=[qa]
    )


def test_tiny_teacher_student_training_loop(tmp_path: Path):
    model = ToyLM(depth=6)
    targets = discover_target_shapes(
        model, [2, 3, 4, 5], ["q_proj", "v_proj", "o_proj", "down_proj"]
    )
    encoder = SharedAtomQueryEncoder(model, 8, 16, 4)
    compiler = AtomMemoryCompiler(targets, input_dim=16, latent_dim=16, key_dim=16, rank=2)
    router = TwoStageRouter(embedding_dim=16, key_dim=16, stage1_top_k=32)
    composer = MemoryComposer()
    injector = DynamicLoRAInjector(model, targets, rank=2, alpha=4)
    system = TeacherStudentSystem(model, encoder, compiler, router, composer, injector)
    config = load_config(Path(__file__).parents[1] / "configs/resource_constrained.yaml")
    config["training"]["max_epochs"] = 1
    config["training"]["gradient_accumulation_steps"] = 1
    result = run_training(
        system, ToyTokenizer(), [_record("train"), _record("validation")], config, tmp_path
    )
    assert result["steps"] == 1
    assert (tmp_path / "last.pt").exists()
    assert torch.isfinite(torch.tensor(result["best_validation_lm"]))

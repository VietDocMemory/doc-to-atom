from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from typing import Callable

from openai import OpenAI

from .models import AnnotationMeta, QuestionAnnotation, StructuredRecord
from .parser import parse_atoms_xml, parse_question_roles_xml
from .prompts import DATASET_ADAPTERS, DECOMPOSITION_SYSTEM, IRRELEVANT_SYSTEM, QUESTION_SYSTEM
from .provenance import RawProviderResponse
from .question_join import (
    QuestionJoinError,
    build_authoritative_questions,
    join_question_roles,
)
from .validation import validate_record


@dataclass
class MiniMaxConfig:
    api_key: str
    base_url: str
    model: str = "MiniMax-M2.5"
    temperature: float = 0.0  # ASSUMPTION: paper does not publish decoding settings.
    max_tokens: int = 8192  # ASSUMPTION.
    irrelevant_probe_count: int = 2  # ASSUMPTION.

    @classmethod
    def from_env(cls) -> MiniMaxConfig:
        return cls(
            api_key=os.environ["MINIMAX_API_KEY"],
            base_url=os.environ["MINIMAX_BASE_URL"],
            model=os.getenv("MINIMAX_MODEL", "MiniMax-M2.5"),
        )


@dataclass
class OpenAIConfig:
    api_key: str
    model: str = "gpt-4.1-mini"
    temperature: float = 0.0
    max_tokens: int = 8192
    irrelevant_probe_count: int = 2
    decomposition_retries: int = 1  # REPRODUCTION_DEVIATION: provider robustness retry.

    @classmethod
    def from_env(cls) -> OpenAIConfig:
        return cls(
            api_key=os.environ["OPENAI_API_KEY"],
            model=os.getenv("OPENAI_ANNOTATION_MODEL", "gpt-4.1-mini"),
        )


class ThreeStageXMLAtomizer:
    def __init__(
        self, config: MiniMaxConfig | OpenAIConfig, provider: str, base_url: str | None = None
    ):
        self.config = config
        self.provider = provider
        self.resolved_model = config.model
        self.last_errors: list[str] = []
        self.last_raw_responses: list[RawProviderResponse] = []
        self.response_sink: Callable[[RawProviderResponse], None] | None = None
        self._active_record_id = ""
        self._stage_attempts: Counter[str] = Counter()
        self.client = OpenAI(api_key=config.api_key, base_url=base_url)

    def _request(self, system: str, user: str) -> str:
        response = self.client.chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        self.resolved_model = response.model
        return response.choices[0].message.content or ""

    def _begin_record(self, record_id: str) -> None:
        self._active_record_id = record_id
        self._stage_attempts = Counter()
        self.last_raw_responses = []

    def _call(self, stage: str, system: str, user: str) -> str:
        content = self._request(system, user)
        self._stage_attempts[stage] += 1
        captured = RawProviderResponse.capture(
            record_id=self._active_record_id,
            stage=stage,
            attempt=self._stage_attempts[stage],
            provider=self.provider,
            model=self.resolved_model,
            content=content,
        )
        self.last_raw_responses.append(captured)
        if self.response_sink is not None:
            self.response_sink(captured)
        return content

    @staticmethod
    def _span_is_verbatim(span: str, context: str) -> bool:
        normalize = lambda value: re.sub(r"\s+", " ", value).strip()
        return normalize(span) in normalize(context)

    @staticmethod
    def _document_xml(context: str) -> str:
        root = ET.Element("document")
        root.text = context
        return ET.tostring(root, encoding="unicode")

    @staticmethod
    def _annotation_input_xml(
        atoms, questions: tuple[QuestionAnnotation, ...]
    ) -> tuple[str, str]:
        atom_root = ET.Element("atoms")
        for atom in atoms:
            node = ET.SubElement(atom_root, "atom", {"id": atom.atom_id})
            ET.SubElement(node, "content").text = atom.content
            ET.SubElement(node, "retrieval_text").text = atom.retrieval_text

        question_root = ET.Element("questions")
        for qa in questions:
            node = ET.SubElement(
                question_root,
                "question",
                {
                    "id": qa.question_id,
                    "is_irrelevant": str(qa.is_irrelevant).lower(),
                    "generated_irrelevant": str(qa.is_generated_irrelevant).lower(),
                },
            )
            ET.SubElement(node, "text").text = qa.question
            answers = ET.SubElement(node, "source_answers", {"read_only": "true"})
            for answer in qa.answers:
                ET.SubElement(answers, "answer").text = answer
        return (
            ET.tostring(atom_root, encoding="unicode"),
            ET.tostring(question_root, encoding="unicode"),
        )

    def atomize(
        self,
        record_id: str,
        context: str,
        questions: list[QuestionAnnotation],
        dataset: str | None = None,
        split: str | None = None,
    ) -> StructuredRecord | None:
        meta = AnnotationMeta()
        self.last_errors = []
        self._begin_record(record_id)
        adapter = DATASET_ADAPTERS.get((dataset or "").lower(), "")
        try:
            document_xml = self._document_xml(context)
            xml = self._call("decomposition", DECOMPOSITION_SYSTEM, f"{adapter}\n{document_xml}")
            atoms, warnings = parse_atoms_xml(xml)
            invalid_spans = [
                atom.source_span
                for atom in atoms
                if not self._span_is_verbatim(atom.source_span, context)
            ]
            if invalid_spans and getattr(self.config, "decomposition_retries", 0):
                repair = (
                    "\nYour previous response used non-verbatim source_span values. Regenerate the "
                    "complete atom XML. For each fact that joins non-contiguous clauses, either copy "
                    "the entire contiguous source sentence or split it into separate atoms. Never use "
                    "ellipsis. Invalid previous spans:\n- " + "\n- ".join(invalid_spans)
                )
                xml = self._call(
                    "decomposition",
                    DECOMPOSITION_SYSTEM + repair,
                    f"{adapter}\n{document_xml}",
                )
                atoms, warnings = parse_atoms_xml(xml)
            meta.warnings.extend(warnings)
            if not atoms:
                meta.errors.append("context decomposition returned no valid atoms")
                self.last_errors = list(meta.errors)
                return None  # Appendix B: discard the whole sample.
        except Exception as exc:  # noqa: BLE001 -- Appendix B requires per-record recovery.
            meta.decomposition_parse_status = "failed"
            meta.errors.append(str(exc))
            self.last_errors = list(meta.errors)
            return None

        generated_probes: list[QuestionAnnotation] = []
        try:
            xml = self._call(
                "irrelevant_probe",
                IRRELEVANT_SYSTEM,
                f"Generate {self.config.irrelevant_probe_count} probes.\n{document_xml}",
            )
            root = ET.fromstring(xml)
            for index, node in enumerate(root.iter("question")):
                if node.text:
                    generated_probes.append(
                        QuestionAnnotation(
                            question_id=f"generated_irrelevant_{index}",
                            question=node.text.strip(),
                            answers=[],
                            answer_provenance="generated_irrelevant",
                            question_type="irrelevant",
                            is_irrelevant=True,
                            is_generated_irrelevant=True,
                        )
                    )
        except Exception as exc:  # noqa: BLE001 -- optional probes must not drop the record.
            meta.irrelevant_parse_status = "failed_ignored"
            meta.warnings.append(f"irrelevant probe generation failed: {exc}")

        try:
            authoritative = build_authoritative_questions(questions, generated_probes)
            atom_xml, question_xml = self._annotation_input_xml(atoms, authoritative)
            xml = self._call(
                "question_annotation",
                QUESTION_SYSTEM,
                f"{adapter}\n{atom_xml}\n{question_xml}",
            )
            roles, notices = parse_question_roles_xml(xml)
            for notice in notices:
                message = f"{notice.code}: {notice.message}"
                if notice.severity == "error":
                    meta.errors.append(message)
                else:
                    meta.warnings.append(message)
            if any(notice.severity == "error" for notice in notices):
                raise QuestionJoinError("provider question annotation failed contract parsing")
            annotated = join_question_roles(authoritative, roles)
        except Exception as exc:  # noqa: BLE001 -- reject incomplete/corrupt QA records.
            meta.question_parse_status = "failed"
            meta.errors.append(f"question annotation failed: {exc}")
            self.last_errors = list(meta.errors)
            return None

        record = StructuredRecord(
            record_id=record_id,
            dataset=dataset,
            split=split,
            context=context,
            atoms=atoms,
            qa_pairs=annotated,
            annotation_meta=meta,
        )
        if self.provider == "openai":
            record.annotation_meta.prompt_provenance = (
                "RECONSTRUCTED_PROMPT_NOT_OFFICIAL; REPRODUCTION_DEVIATION: "
                f"OpenAI replaces MiniMax-M2.5; model={self.resolved_model}"
            )
        return validate_record(record)


class MiniMaxAtomizer(ThreeStageXMLAtomizer):
    def __init__(self, config: MiniMaxConfig):
        super().__init__(config, provider="minimax", base_url=config.base_url)


class OpenAIAtomizer(ThreeStageXMLAtomizer):
    """REPRODUCTION_DEVIATION: OpenAI replaces the paper's MiniMax-M2.5 annotator."""

    def __init__(self, config: OpenAIConfig):
        super().__init__(config, provider="openai")

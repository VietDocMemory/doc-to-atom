from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from openai import OpenAI

from .models import AnnotationMeta, QuestionAnnotation, StructuredRecord
from .parser import parse_atoms_xml, parse_questions_xml
from .prompts import DATASET_ADAPTERS, DECOMPOSITION_SYSTEM, IRRELEVANT_SYSTEM, QUESTION_SYSTEM
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


class MiniMaxAtomizer:
    def __init__(self, config: MiniMaxConfig):
        self.config = config
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)

    def _call(self, system: str, user: str) -> str:
        response = self.client.chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return response.choices[0].message.content or ""

    def atomize(
        self,
        record_id: str,
        context: str,
        questions: list[QuestionAnnotation],
        dataset: str | None = None,
        split: str | None = None,
    ) -> StructuredRecord | None:
        meta = AnnotationMeta()
        adapter = DATASET_ADAPTERS.get((dataset or "").lower(), "")
        try:
            xml = self._call(DECOMPOSITION_SYSTEM, f"{adapter}\n<document>{context}</document>")
            atoms, warnings = parse_atoms_xml(xml)
            meta.warnings.extend(warnings)
            if not atoms:
                meta.errors.append("context decomposition returned no valid atoms")
                return None  # Appendix B: discard the whole sample.
        except Exception as exc:  # noqa: BLE001 -- Appendix B requires per-record recovery.
            meta.decomposition_parse_status = "failed"
            meta.errors.append(str(exc))
            return None

        augmented = list(questions)
        try:
            xml = self._call(
                IRRELEVANT_SYSTEM,
                f"Generate {self.config.irrelevant_probe_count} probes.\n<document>{context}</document>",
            )
            root = ET.fromstring(xml)
            for index, node in enumerate(root.iter("question")):
                if node.text:
                    augmented.append(
                        QuestionAnnotation(
                            question_id=f"generated_irrelevant_{index}",
                            question=node.text.strip(),
                            question_type="irrelevant",
                            is_irrelevant=True,
                            is_generated_irrelevant=True,
                        )
                    )
        except Exception as exc:  # noqa: BLE001 -- optional probes must not drop the record.
            meta.irrelevant_parse_status = "failed_ignored"
            meta.warnings.append(f"irrelevant probe generation failed: {exc}")

        atom_xml = "\n".join(
            f'<atom id="{a.atom_id}"><content>{a.content}</content><retrieval_text>{a.retrieval_text}</retrieval_text></atom>'
            for a in atoms
        )
        question_xml = "\n".join(
            f'<question id="{q.question_id}" generated_irrelevant="{str(q.is_generated_irrelevant).lower()}">{q.question}</question>'
            for q in augmented
        )
        try:
            xml = self._call(
                QUESTION_SYSTEM,
                f"{adapter}\n<atoms>{atom_xml}</atoms>\n<questions>{question_xml}</questions>",
            )
            annotated, warnings = parse_questions_xml(xml)
            meta.warnings.extend(warnings)
        except Exception as exc:  # noqa: BLE001 -- preserve atom-only records on annotation failure.
            meta.question_parse_status = "failed_atom_only"
            meta.warnings.append(f"question annotation failed: {exc}")
            annotated = []  # Appendix B: keep atom-only record.

        record = StructuredRecord(
            record_id=record_id,
            dataset=dataset,
            split=split,
            context=context,
            atoms=atoms,
            qa_pairs=annotated,
            annotation_meta=meta,
        )
        return validate_record(record)


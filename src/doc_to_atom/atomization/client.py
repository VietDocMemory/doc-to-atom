from __future__ import annotations

import os
import re
import time
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

import httpx
from openai import APIConnectionError, APIStatusError, OpenAI

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


DEFAULT_MINIMAX_MODEL = "MiniMax-M3.1-Flash"
DEFAULT_REASONING_EFFORT = "low"  # Provisional choice from the paired annotation pilot.
MINIMAX_BASE_URL = "https://api.minimax.io/v1"
MINIMAX_CHAT_ENDPOINT = f"{MINIMAX_BASE_URL}/chat/completions"


@dataclass
class MiniMaxConfig:
    api_key: str = field(repr=False)
    model: str = DEFAULT_MINIMAX_MODEL
    max_completion_tokens: int = 8192  # Budget assumption, includes thinking tokens.
    irrelevant_probe_count: int = 2  # ASSUMPTION.
    reasoning_effort: str = DEFAULT_REASONING_EFFORT
    timeout_seconds: float = 180.0

    def __post_init__(self) -> None:
        if not self.api_key.strip():
            raise ValueError("MINIMAX_API_KEY or OPENAI_API_KEY must not be empty")
        if not self.model.startswith("MiniMax-"):
            raise ValueError("annotation requires a MiniMax model ID")
        if self.reasoning_effort not in {"low", "medium", "high", "xhigh", "max"}:
            raise ValueError("invalid MiniMax reasoning_effort")
        if self.max_completion_tokens < 1 or self.timeout_seconds <= 0:
            raise ValueError("completion budget and timeout must be positive")

    @classmethod
    def from_env(cls) -> MiniMaxConfig:
        # This pipeline remains MiniMax-only; never send its key to another host.
        if os.environ.get("OPENAI_BASE_URL", MINIMAX_BASE_URL).rstrip("/") != MINIMAX_BASE_URL:
            raise ValueError(f"OPENAI_BASE_URL must be {MINIMAX_BASE_URL}")
        return cls(
            api_key=os.environ.get("MINIMAX_API_KEY") or os.environ.get("OPENAI_API_KEY", ""),
        )


class MiniMaxAPIError(RuntimeError):
    """A transport/provider failure; never silently spend on subsequent records."""


class AnnotationResponseError(MiniMaxAPIError):
    """Recoverable annotation failure, distinct from fatal infrastructure errors."""

    def __init__(self, message: str, partial_content: str = ""):
        super().__init__(message)
        self.partial_content = partial_content


class ThreeStageXMLAtomizer:
    def __init__(self, config: MiniMaxConfig):
        self.config = config
        self.provider = "minimax"
        self.resolved_model = config.model
        self.last_errors: list[str] = []
        self.last_raw_responses: list[RawProviderResponse] = []
        self.response_sink: Callable[[RawProviderResponse], None] | None = None
        self._active_record_id = ""
        self._stage_attempts: Counter[str] = Counter()
        self._response_metadata: dict = {}
        self._response_error: str | None = None
        self._response_partial = False
        self._response_fatal = False
        self.last_annotation_meta = AnnotationMeta()

    def _request(self, system: str, user: str) -> str:
        # Use the official OpenAI SDK, with MiniMax extensions in extra_body.
        parameters = {
            "model": self.config.model,
            "max_completion_tokens": self.config.max_completion_tokens,
            "reasoning_split": True,
        }
        if self.config.model == "MiniMax-M3.1-Flash":
            parameters["reasoning_effort"] = self.config.reasoning_effort
        started = time.perf_counter()
        try:
            with OpenAI(
                api_key=self.config.api_key,
                base_url=MINIMAX_BASE_URL,
                max_retries=0,
                timeout=self.config.timeout_seconds,
                http_client=httpx.Client(follow_redirects=False),
            ) as client:
                raw_response = client.chat.completions.with_raw_response.create(
                    model=self.config.model,
                    max_completion_tokens=self.config.max_completion_tokens,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    extra_body={"reasoning_split": True},
                    **({"reasoning_effort": self.config.reasoning_effort}
                       if self.config.model == "MiniMax-M3.1-Flash" else {}),
                )
                # Preserve exact JSON, including reasoning_content and tool_calls,
                # rather than rebuilding a response from just the visible text.
                response = raw_response.http_response
        except APIStatusError as error:
            response = error.response
        except APIConnectionError:
            raise MiniMaxAPIError("MiniMax connection failed or timed out; no automatic retry") from None
        self._response_metadata = {
            "raw_body": response.text,
            "http_status": response.status_code,
            "requested_model": self.config.model,
            "request_parameters": parameters,
            "latency_ms": (time.perf_counter() - started) * 1000,
        }
        if not response.is_success:
            hint = {
                401: "check MINIMAX_API_KEY or OPENAI_API_KEY",
                403: "check model/account permissions",
                402: "insufficient balance; check MiniMax billing",
                429: "rate limited; retry manually later",
            }.get(response.status_code, "inspect the response sidecar before retrying")
            self._response_error = f"MiniMax HTTP {response.status_code}: {hint}"
            return ""
        try:
            payload = response.json()
            base = payload.get("base_resp") or {}
            if base.get("status_code", 0) != 0:
                self._response_fatal = True
                self._response_error = f"MiniMax provider error {base['status_code']}; inspect response sidecar"
                return ""
            self.resolved_model = payload["model"]
            choice = payload["choices"][0]
            content = choice["message"].get("content") or ""
            if not isinstance(content, str) or not isinstance(self.resolved_model, str):
                raise ValueError("invalid text/model")
            if choice.get("finish_reason") != "stop" or not content.strip():
                self._response_partial = choice.get("finish_reason") == "length" and bool(content.strip())
                self._response_error = (
                    f"MiniMax incomplete completion ({choice.get('finish_reason')}); "
                    "review max_completion_tokens (includes thinking) and response sidecar"
                )
            return content
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            self._response_error = "MiniMax returned an invalid Chat Completions response; inspect sidecar"
            return ""

    def _begin_record(self, record_id: str) -> None:
        self._active_record_id = record_id
        self.resolved_model = self.config.model
        self._stage_attempts = Counter()
        self.last_raw_responses = []

    def _call(self, stage: str, system: str, user: str) -> str:
        self._response_metadata = {}
        self._response_error = None
        self._response_partial = False
        self._response_fatal = False
        content = self._request(system, user)
        self._stage_attempts[stage] += 1
        captured = RawProviderResponse.capture(
            record_id=self._active_record_id,
            stage=stage,
            attempt=self._stage_attempts[stage],
            provider=self.provider,
            model=self.resolved_model,
            content=content,
            **self._response_metadata,
        )
        self.last_raw_responses.append(captured)
        if self.response_sink is not None:
            self.response_sink(captured)
        if self._response_error:
            status = self._response_metadata.get("http_status")
            # Keep the legacy paid provider fail-closed: malformed completions
            # must not silently lead to more billed downstream requests.
            if (self.provider == "minimax" or self._response_fatal
                    or (status is not None and not 200 <= status < 300)):
                raise MiniMaxAPIError(self._response_error)
            partial = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
            raise AnnotationResponseError(
                self._response_error, partial_content=partial if self._response_partial else "",
            )
        # Older MiniMax models may include inline thinking. Keep raw content
        # intact in provenance and exclude thinking only from the parser input.
        return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)

    def _call_for_stage(self, stage: str, system: str, user: str, meta: AnnotationMeta) -> str:
        try:
            return self._call(stage, system, user)
        except AnnotationResponseError as exc:
            # Reuse only complete XML subtrees. No second inference call or
            # nondeterministic repair is introduced by this recovery path.
            if exc.partial_content:
                meta.warnings.append(f"{stage}: recovering complete XML subtrees from partial output")
                return exc.partial_content
            raise ValueError(str(exc)) from exc

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
            if not atom.span_valid:
                continue
            node = ET.SubElement(atom_root, "atom", {"id": atom.atom_id})
            ET.SubElement(node, "content").text = atom.content
            ET.SubElement(node, "source_span").text = atom.source_span
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
        self.last_annotation_meta = meta
        self.last_errors = []
        self._begin_record(record_id)
        self._stamp_provenance(meta)
        adapter = DATASET_ADAPTERS.get((dataset or "").lower(), "")
        try:
            document_xml = self._document_xml(context)
            xml = self._call_for_stage(
                "decomposition", DECOMPOSITION_SYSTEM, f"{adapter}\n{document_xml}", meta,
            )
            atoms, warnings = parse_atoms_xml(xml)
            meta.warnings.extend(warnings)
            if not atoms:
                meta.decomposition_parse_status = "failed"
                meta.errors.append("context decomposition returned no valid atoms")
                self.last_errors = list(meta.errors)
                return None  # Appendix B: discard the whole sample.
            # Appendix B.1: validate before any question-to-atom annotation.
            # Retain invalid atoms/flags in the artifact for quality inspection,
            # but never present them as grounded evidence in the annotation bank.
            bank = validate_record(StructuredRecord(
                record_id=record_id, dataset=dataset, split=split, context=context,
                atoms=atoms, qa_pairs=[], annotation_meta=meta,
            ))
            atoms, meta = bank.atoms, bank.annotation_meta
            self.last_annotation_meta = meta
            if not any(atom.span_valid for atom in atoms):
                meta.decomposition_parse_status = "failed"
                meta.errors.append("context decomposition returned no source-grounded atoms")
                self.last_errors = list(meta.errors)
                return None
        except MiniMaxAPIError:
            raise
        except Exception as exc:  # noqa: BLE001 -- Appendix B requires per-record recovery.
            meta.decomposition_parse_status = "failed"
            meta.errors.append(str(exc))
            self.last_errors = list(meta.errors)
            return None

        generated_probes: list[QuestionAnnotation] = []
        try:
            xml = self._call_for_stage(
                "irrelevant_probe",
                IRRELEVANT_SYSTEM,
                f"Generate {self.config.irrelevant_probe_count} probes.\n{document_xml}",
                meta,
            )
            root = ET.fromstring(xml)
            if root.tag != "irrelevant_questions":
                raise ValueError("expected irrelevant_questions XML root")
            for index, node in enumerate(root.iter("question")):
                if list(node):
                    raise ValueError("irrelevant probe must be text-only")
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
        except MiniMaxAPIError:
            raise
        except Exception as exc:  # noqa: BLE001 -- optional probes must not drop the record.
            generated_probes = []
            meta.irrelevant_parse_status = "failed_ignored"
            meta.warnings.append(f"irrelevant probe generation failed: {exc}")

        try:
            authoritative = build_authoritative_questions(questions, generated_probes)
            atom_xml, question_xml = self._annotation_input_xml(atoms, authoritative)
            xml = self._call_for_stage(
                "question_annotation",
                QUESTION_SYSTEM,
                f"{adapter}\n{atom_xml}\n{question_xml}",
                meta,
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
            role_by_id = {role.question_id: role for role in roles}
            for qa in annotated:
                role = role_by_id[qa.question_id]
                if qa.is_generated_irrelevant and (
                    role.question_type not in (None, "irrelevant") or role.is_irrelevant is False
                ):
                    meta.warnings.append(
                        f"{qa.question_id}: generated probe forced irrelevant (Appendix B.2)"
                    )
        except MiniMaxAPIError:
            raise
        except Exception as exc:  # noqa: BLE001 -- reject incomplete/corrupt QA records.
            meta.question_parse_status = "failed"
            meta.errors.append(f"question annotation failed: {exc}")
            self._stamp_provenance(meta)
            self.last_errors = list(meta.errors)
            # Appendix B.4: keep the validated atom bank, without unreliable
            # question annotations. This record is not a training example.
            return validate_record(StructuredRecord(
                record_id=record_id,
                dataset=dataset,
                split=split,
                context=context,
                atoms=atoms,
                qa_pairs=[],
                annotation_meta=meta,
            ))

        record = StructuredRecord(
            record_id=record_id,
            dataset=dataset,
            split=split,
            context=context,
            atoms=atoms,
            qa_pairs=annotated,
            annotation_meta=meta,
        )
        self._stamp_provenance(record.annotation_meta)
        return validate_record(record)

    def _stamp_provenance(self, meta: AnnotationMeta) -> None:
        meta.prompt_provenance = (
            f"provider=minimax; requested_model={self.config.model}; model={self.resolved_model}"
        )


class MiniMaxAtomizer(ThreeStageXMLAtomizer):
    """Public MiniMax-only entry point for the three-stage annotation pipeline."""

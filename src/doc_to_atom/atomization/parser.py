from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Iterable

from .models import Atom, ParseNotice, QuestionRoleAnnotation, Relation


def _recover_elements(xml: str, tag: str) -> Iterable[ET.Element]:
    """Yield well-formed subtrees even when the outer response is truncated."""
    try:
        root = ET.fromstring(xml)
        yield from root.iter(tag)
        return
    except ET.ParseError:
        pattern = re.compile(rf"<{tag}\b[^>]*>.*?</{tag}>", re.DOTALL)
        for match in pattern.finditer(xml):
            try:
                yield ET.fromstring(match.group(0))
            except ET.ParseError:
                continue


def _text(element: ET.Element, name: str, default: str = "") -> str:
    node = element.find(name)
    return default if node is None or node.text is None else node.text.strip()


def _value(element: ET.Element, name: str, default: str = "") -> str:
    """Read equivalent XML attribute or child-element scalar forms."""
    child = _text(element, name)
    return child if child else (element.get(name) or default).strip()


def _bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def _ids(element: ET.Element, name: str) -> list[str]:
    node = element.find(name)
    if node is None:
        return [item.strip() for item in element.get(name, "").split(",") if item.strip()]
    explicit = [child.text.strip() for child in node.findall("atom_id") if child.text]
    if explicit:
        return explicit
    return [item.strip() for item in (node.text or "").split(",") if item.strip()]


def parse_atoms_xml(xml: str) -> tuple[list[Atom], list[str]]:
    atoms: list[Atom] = []
    warnings: list[str] = []
    for index, node in enumerate(_recover_elements(xml, "atom")):
        try:
            abstraction_level = _value(node, "abstraction_level")
            aliases = {
                "basic",
                "detailed",
                "fact_claim",
                "entity_attribute",
                "event_relation",
                "process_step",
                "evidence_fragment",
            }
            if (
                abstraction_level not in {"abstract", "evidence", "hybrid"}
                and abstraction_level in aliases
            ):
                warnings.append(
                    f"atom[{index}] normalized abstraction_level={abstraction_level!r} to 'evidence'"
                )
                abstraction_level = "evidence"
            relations = []
            relation_root = node.find("relations")
            if relation_root is not None:
                for rel in relation_root.findall("relation"):
                    relations.append(
                        Relation(
                            relation_type=rel.get("type") or _text(rel, "relation_type"),
                            target_atom_id=rel.get("target") or _text(rel, "target_atom_id"),
                        )
                    )
            atoms.append(
                Atom(
                    atom_id=node.get("id") or _text(node, "atom_id") or f"atom_{index}",
                    semantic_type=_text(node, "semantic_type"),
                    content=_text(node, "content"),
                    source_span=_text(node, "source_span"),
                    retrieval_text=_text(node, "retrieval_text"),
                    is_answer_bearing=_bool(_text(node, "is_answer_bearing")),
                    abstraction_level=abstraction_level,
                    conflict_group=_text(node, "conflict_group") or None,
                    confidence=float(_text(node, "confidence", "0")),
                    relations=relations,
                )
            )
        except Exception as exc:  # noqa: BLE001 -- malformed generated elements are isolated.
            warnings.append(f"atom[{index}] dropped: {exc}")
    return atoms, warnings


def parse_question_roles_xml(xml: str) -> tuple[list[QuestionRoleAnnotation], list[ParseNotice]]:
    annotations: list[QuestionRoleAnnotation] = []
    notices: list[ParseNotice] = []
    forbidden = {"answer", "answers", "question", "refusal_target"}
    for index, node in enumerate(_recover_elements(xml, "question_annotation")):
        present = sorted(
            forbidden.intersection(node.attrib).union(
                child.tag for child in list(node) if child.tag in forbidden
            )
        )
        if present:
            notices.append(
                ParseNotice(
                    severity="error",
                    code="FORBIDDEN_ANNOTATION_FIELD",
                    message=f"provider annotation contains forbidden fields: {', '.join(present)}",
                    element_index=index,
                )
            )
            continue
        try:
            question_type = _value(node, "question_type") or None
            annotations.append(
                QuestionRoleAnnotation(
                    question_id=node.get("id") or _value(node, "question_id"),
                    question_type=question_type,
                    gold_atom_ids=tuple(_ids(node, "gold_atom_ids")),
                    supporting_atom_ids=tuple(_ids(node, "supporting_atom_ids")),
                    distractor_atom_ids=tuple(_ids(node, "distractor_atom_ids")),
                )
            )
        except Exception as exc:  # noqa: BLE001 -- malformed generated elements are isolated.
            notices.append(
                ParseNotice(
                    severity="error",
                    code="QUESTION_ANNOTATION_PARSE_ERROR",
                    message=f"question[{index}] dropped: {exc}",
                    element_index=index,
                )
            )
    return annotations, notices


def parse_questions_xml(xml: str) -> tuple[list[QuestionRoleAnnotation], list[ParseNotice]]:
    """Deprecated compatibility name for the schema-v2 role-only parser."""
    return parse_question_roles_xml(xml)

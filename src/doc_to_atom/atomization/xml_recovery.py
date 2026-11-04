"""Narrow, auditable XML syntax recovery. Never changes question/answer text."""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET


def recover_annotation_xml(text: str, root_tag: str) -> tuple[str, list[dict]]:
    if root_tag not in {"irrelevant_questions", "question_annotations"}:
        raise ValueError("Unsupported annotation XML root")
    original = text
    events = []

    def change(code: str, updated: str) -> None:
        nonlocal text
        if updated != text:
            events.append({"code": code, "before": text, "after": updated})
            text = updated

    if text.startswith(root_tag + ">"):
        change("INSERT_ROOT_OPEN_DELIMITER", "<" + text)
    # Only the exact line-initial opening-tag defect seen in saved annotations.
    change("INSERT_QUESTION_OPEN_DELIMITER", re.sub(
        r"(?m)^(\s*)question_annotation>", r"\1<question_annotation>", text))
    # XML escaping changes serialization, not the decoded question text.
    change("ESCAPE_BARE_AMPERSAND", re.sub(
        r"&(?!amp;|lt;|gt;|quot;|apos;|#\d+;|#x[0-9a-fA-F]+;)", "&amp;", text))
    if root_tag == "irrelevant_questions":
        while re.search(r"</question>\s*</question>", text):
            change("REMOVE_DUPLICATE_QUESTION_CLOSE", re.sub(
                r"(</question>)\s*</question>", r"\1", text))
    root = ET.fromstring(text)  # Refuse any other defect, never silently drop a subtree.
    if root.tag != root_tag:
        raise ValueError("Unexpected XML root")
    if root_tag == "irrelevant_questions":
        if not list(root) or (root.text or "").strip():
            raise ValueError("Empty or mixed-content probe output")
        for question in root:
            if (question.tag != "question" or list(question) or not (question.text or "").strip()
                    or (question.tail or "").strip()):
                raise ValueError("Probe output is not a list of complete text-only questions")
    if not events and original != text:
        raise AssertionError("Unlogged XML edit")
    return text, events

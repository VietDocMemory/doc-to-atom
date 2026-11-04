import xml.etree.ElementTree as ET

import pytest

from doc_to_atom.atomization.xml_recovery import recover_annotation_xml


def test_bare_ampersands_and_duplicate_close_preserve_decoded_probe_text():
    raw = "<irrelevant_questions><question>What about R&B and H&amp;M?</question></question></irrelevant_questions>"
    derived, events = recover_annotation_xml(raw, "irrelevant_questions")
    assert ET.fromstring(derived)[0].text == "What about R&B and H&M?"
    assert {event["code"] for event in events} == {"ESCAPE_BARE_AMPERSAND", "REMOVE_DUPLICATE_QUESTION_CLOSE"}
    assert raw.endswith("</question></question></irrelevant_questions>")


def test_missing_delimiter_is_repaired_without_dropping_question():
    raw = "<question_annotations>\n question_annotation><question_id>q</question_id></question_annotation></question_annotations>"
    derived, events = recover_annotation_xml(raw, "question_annotations")
    assert len(ET.fromstring(derived)) == 1
    assert events[0]["code"] == "INSERT_QUESTION_OPEN_DELIMITER"


@pytest.mark.parametrize("raw", ["<irrelevant_questions><question>partial", "<other/>",
                                  "<irrelevant_questions>explanation<question>Why?</question></irrelevant_questions>"])
def test_recovery_refuses_ambiguous_or_incomplete_text(raw):
    with pytest.raises((ValueError, ET.ParseError)):
        recover_annotation_xml(raw, "irrelevant_questions")


def test_valid_xml_is_unchanged():
    raw = "<irrelevant_questions><question>Why?</question></irrelevant_questions>"
    assert recover_annotation_xml(raw, "irrelevant_questions") == (raw, [])

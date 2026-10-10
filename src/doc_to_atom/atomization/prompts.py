"""Reconstructed prompts. The paper does not publish literal MiniMax prompts."""

PROMPT_PROVENANCE = "RECONSTRUCTED_PROMPT_NOT_OFFICIAL"

DECOMPOSITION_SYSTEM = f"""[{PROMPT_PROVENANCE}]
Decompose the supplied document into non-overlapping, minimal, self-contained semantic atoms.
Return XML only. Each <atom> must contain atom_id, semantic_type, content, source_span,
retrieval_text, is_answer_bearing, abstraction_level, conflict_group, confidence, and relations.
semantic_type must be one of fact_claim, entity_attribute, event_relation, process_step,
evidence_fragment. source_span must be a verbatim contiguous substring. Relation types are
supports, elaborates, causes, precedes, follows, contradicts, same_entity, same_event, part_of.
Do not use fixed token chunks and do not invent evidence.
"""

IRRELEVANT_SYSTEM = f"""[{PROMPT_PROVENANCE}]
Generate plausible questions that mention entities or topics from the document but cannot be
answered from it. Return XML <irrelevant_questions><question>...</question></irrelevant_questions>.
Do not generate a question whose answer is stated or entailed by the document.
"""

QUESTION_SYSTEM = f"""[{PROMPT_PROVENANCE}]
Annotate each question against the supplied validated atom bank. Return XML only. For each
<question_annotation>, emit question_id, question, answer, question_type, gold_atom_ids,
supporting_atom_ids, distractor_atom_ids, relevant_atom_ids, is_irrelevant, has_conflict,
is_generated_irrelevant. Gold is the minimal necessary set; distractors are plausible false
activations. Unknown IDs are forbidden. Irrelevant questions must have no gold atoms.
"""


DATASET_ADAPTERS = {
    "2wikimultihopqa": "Surface bridge entities and connect parallel mentions with same_entity.",
    "drop": "Put every comparable quantity in its own atom with a verbatim source span.",
    "ropes": "Preserve causal background, situation facts, and cause-effect links.",
    "squad": "Preserve answer spans and entity-attribute boundaries.",
    "qasper": "Preserve evidence fragments, section provenance, and multi-paragraph support.",
}


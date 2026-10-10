"""Reconstructed prompts. The paper does not publish literal MiniMax prompts."""

PROMPT_PROVENANCE = "RECONSTRUCTED_PROMPT_NOT_OFFICIAL"

DECOMPOSITION_SYSTEM = f"""[{PROMPT_PROVENANCE}]
Decompose the supplied document into non-overlapping, minimal, self-contained semantic atoms.
Return one <atoms> XML root and no markdown. Each <atom> must contain atom_id, semantic_type,
content, source_span, retrieval_text, is_answer_bearing, abstraction_level, conflict_group,
confidence, and relations. source_span must be copied as one verbatim contiguous substring:
never insert `...`, an ellipsis character, brackets, paraphrases, or omitted words. Do not use
fixed token chunks and do not invent evidence.

Classify every atom on TWO INDEPENDENT AXES. Never copy a label from one axis to the other.

Axis 1 — semantic_type describes WHAT the atom means. It must be exactly one of:
- fact_claim: a standalone proposition asserted as true, not primarily an entity property,
  explicit event-to-event/entity relation, ordered instruction, or raw evidentiary quotation.
- entity_attribute: a named entity or object paired with a property, value, identity, location,
  quantity, title, or descriptive attribute.
- event_relation: an event or state explicitly linked to another event/entity by time, cause,
  dependency, comparison, participation, or another semantic relation.
- process_step: one ordered action, instruction, or stage in a procedure or process.
- evidence_fragment: a quotation, observation, measurement, or textual fragment whose primary
  role is to serve as evidence rather than state a normalized claim.

Axis 2 — abstraction_level describes HOW content relates to source_span. It must be exactly:
- evidence: content closely preserves one explicit source statement or quotation.
- abstract: content is a generalized or compressed inference spanning source wording.
- hybrid: content combines a normalized/generalized statement with essential concrete evidence.

The only legal abstraction_level strings are abstract, evidence, hybrid. Values such as
fact_claim, entity_attribute, basic, or detailed are INVALID in abstraction_level. Before
returning XML, check every atom independently against both enum lists.

Example:
<atom><atom_id>atom_1</atom_id><semantic_type>entity_attribute</semantic_type>
<content>Paris is the capital of France.</content>
<source_span>Paris is the capital of France.</source_span>
<retrieval_text>Paris capital France</retrieval_text><is_answer_bearing>true</is_answer_bearing>
<abstraction_level>evidence</abstraction_level><conflict_group/><confidence>0.99</confidence>
<relations/></atom>

Relation types are supports, elaborates, causes, precedes, follows, contradicts, same_entity,
same_event, part_of. Use <relations><relation type="supports" target="atom_1"/></relations>;
use empty <relations/> and <conflict_group/> when absent. atom_id values must be unique.
confidence must be a number from 0 through 1.
"""

IRRELEVANT_SYSTEM = f"""[{PROMPT_PROVENANCE}]
Generate plausible questions that mention entities or topics from the document but cannot be
answered from it. Return XML <irrelevant_questions><question>...</question></irrelevant_questions>.
Do not generate a question whose answer is stated or entailed by the document.
"""

QUESTION_SYSTEM = f"""[{PROMPT_PROVENANCE}]
Annotate each question against the supplied validated atom bank. Return one
<question_annotations> XML root and no markdown. For each
<question_annotation>, emit question_id, question, answer, question_type, gold_atom_ids,
supporting_atom_ids, distractor_atom_ids, relevant_atom_ids, is_irrelevant, has_conflict,
is_generated_irrelevant. Gold is the minimal necessary set; distractors are plausible false
activations. Unknown IDs are forbidden. Irrelevant questions must have no gold atoms.
Represent each ID list as repeated <atom_id>...</atom_id> children. Copy question_id from
the input question id attribute and preserve generated_irrelevant in is_generated_irrelevant.
question_type must be exactly one of single_hop, multi_hop, comparison, temporal, causal,
procedural, aggregation, field_lookup, evidence_grounded, irrelevant.
"""


DATASET_ADAPTERS = {
    "2wikimultihopqa": "Surface bridge entities and connect parallel mentions with same_entity.",
    "drop": "Put every comparable quantity in its own atom with a verbatim source span.",
    "ropes": "Preserve causal background, situation facts, and cause-effect links.",
    "squad": "Preserve answer spans and entity-attribute boundaries.",
    "qasper": "Preserve evidence fragments, section provenance, and multi-paragraph support.",
}

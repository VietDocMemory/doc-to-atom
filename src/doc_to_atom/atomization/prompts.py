"""XML annotation prompts for the Doc-to-Atom pipeline."""

DECOMPOSITION_SYSTEM = """Decompose the supplied document into non-overlapping, minimal, self-contained semantic atoms.
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

IRRELEVANT_SYSTEM = """Generate plausible questions that mention entities or topics from the document but cannot be
answered from it. Return XML <irrelevant_questions><question>...</question></irrelevant_questions>.
Do not generate a question whose answer is stated or entailed by the document.
"""

QUESTION_SYSTEM = """Annotate each question against the supplied validated atom bank. Return one
<question_annotations> XML root and no markdown. For each
<question_annotation>, emit only question_id, question_type, gold_atom_ids,
supporting_atom_ids, and distractor_atom_ids. Gold is the minimal necessary set; distractors
are plausible false activations. Unknown IDs are forbidden. Irrelevant questions must have no
gold or supporting atoms. Input question text and source answers are read-only evidence: never
copy, rewrite, normalize, paraphrase, or emit them. Never emit answer, answers, question,
refusal_target, relevant_atom_ids, is_irrelevant, has_conflict, or is_generated_irrelevant.
Represent each ID list as repeated <atom_id>...</atom_id> children. Copy question_id from
the input question id attribute exactly.
question_type must be exactly one of single_hop, multi_hop, comparison, temporal, causal,
procedural, aggregation, field_lookup, evidence_grounded, irrelevant.

Assign roles independently for EACH question. The three ID lists must be pairwise disjoint:
- GOLD: the smallest sufficient evidence set for the source answer. Keep every required
  bridge/comparison atom for multi-hop questions. If removing an atom breaks the answer's
  evidence chain, that atom is gold, NOT supporting.
- SUPPORTING: optional background that helps explain the answer but is unnecessary once
  gold is present. Do not repeat answer-bearing gold atoms here. An empty list is normal,
  especially for a direct single-hop lookup.
- DISTRACTOR: plausible but incorrect evidence for this question, never gold or supporting.
  Not every unused atom is a distractor. Empty distractor lists are allowed.
Do not invent extra gold or supporting atoms to fill a list. Do not label a source-answerable
question irrelevant merely to avoid choosing gold. For unanswerable questions use
question_type=irrelevant and empty gold/supporting lists.

Example: atom_1 states Paris is France's capital; atom_2 states Paris is on the Seine;
atom_3 states Lyon is a city in France. For "What is France's capital?", emit:
<question_annotation><question_id>q1</question_id><question_type>single_hop</question_type>
<gold_atom_ids><atom_id>atom_1</atom_id></gold_atom_ids><supporting_atom_ids/>
<distractor_atom_ids><atom_id>atom_3</atom_id></distractor_atom_ids></question_annotation>
atom_1 must NOT also appear in supporting_atom_ids. atom_2 need not receive any role.
Before emitting XML, silently check: no duplicate IDs within a list; gold intersects
supporting is empty; gold intersects distractor is empty; supporting intersects distractor
is empty; all IDs exist in the input bank; every input question_id occurs exactly once.
"""


DATASET_ADAPTERS = {
    "2wikimultihopqa": "Surface bridge entities and connect parallel mentions with same_entity.",
    "drop": "Put every comparable quantity in its own atom with a verbatim source span.",
    "ropes": "Preserve causal background, situation facts, and cause-effect links.",
    "squad": "Preserve answer spans and entity-attribute boundaries.",
    "qasper": "Preserve evidence fragments, section provenance, and multi-paragraph support.",
}

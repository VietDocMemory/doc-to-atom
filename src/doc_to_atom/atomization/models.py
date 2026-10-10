from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

SemanticType = Literal[
    "fact_claim",
    "entity_attribute",
    "event_relation",
    "process_step",
    "evidence_fragment",
]
AbstractionLevel = Literal["abstract", "evidence", "hybrid"]
RelationType = Literal[
    "supports",
    "elaborates",
    "causes",
    "precedes",
    "follows",
    "contradicts",
    "same_entity",
    "same_event",
    "part_of",
]
QuestionType = Literal[
    "single_hop",
    "multi_hop",
    "comparison",
    "temporal",
    "causal",
    "procedural",
    "aggregation",
    "field_lookup",
    "evidence_grounded",
    "irrelevant",
]


class Relation(BaseModel):
    relation_type: RelationType
    target_atom_id: str


class Atom(BaseModel):
    atom_id: str
    semantic_type: SemanticType
    content: str
    source_span: str
    retrieval_text: str
    is_answer_bearing: bool
    abstraction_level: AbstractionLevel
    conflict_group: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    relations: list[Relation] = Field(default_factory=list)
    span_valid: bool = True

    @field_validator("atom_id", "content", "source_span", "retrieval_text")
    @classmethod
    def nonempty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value


class QuestionAnnotation(BaseModel):
    question_id: str
    question: str
    answer: str | list[str] | None = None
    question_type: QuestionType
    gold_atom_ids: list[str] = Field(default_factory=list)
    supporting_atom_ids: list[str] = Field(default_factory=list)
    distractor_atom_ids: list[str] = Field(default_factory=list)
    relevant_atom_ids: list[str] = Field(default_factory=list)
    is_irrelevant: bool = False
    has_conflict: bool = False
    is_generated_irrelevant: bool = False
    annotation_warnings: list[str] = Field(default_factory=list)


class AnnotationMeta(BaseModel):
    mode: str = "three_stage_xml"
    decomposition_parse_status: str = "ok"
    irrelevant_parse_status: str = "ok"
    question_parse_status: str = "ok"
    num_atoms: int = 0
    num_questions: int = 0
    num_irrelevant_generated: int = 0
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    prompt_provenance: str = "RECONSTRUCTED_PROMPT_NOT_OFFICIAL"


class StructuredRecord(BaseModel):
    record_id: str
    dataset: str | None = None
    split: str | None = None
    context: str
    atoms: list[Atom]
    qa_pairs: list[QuestionAnnotation] = Field(default_factory=list)
    annotation_meta: AnnotationMeta = Field(default_factory=AnnotationMeta)


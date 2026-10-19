from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ATOMIZATION_SCHEMA_VERSION = 2

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


class TransformationEvent(BaseModel):
    code: str
    stage: str
    message: str
    question_id: str | None = None
    atom_id: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class ParseNotice(BaseModel):
    severity: Literal["warning", "error"]
    code: str
    message: str
    element_index: int | None = None


class QuestionRoleAnnotation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str
    question_type: QuestionType | None = None
    gold_atom_ids: tuple[str, ...] = Field(default_factory=tuple)
    supporting_atom_ids: tuple[str, ...] = Field(default_factory=tuple)
    distractor_atom_ids: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator("question_id")
    @classmethod
    def nonempty_question_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question_id must not be empty")
        return value.strip()


class QuestionAnnotation(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    question_id: str = Field(frozen=True)
    question: str = Field(frozen=True)
    answers: tuple[str, ...] = Field(default_factory=tuple, frozen=True)
    answer_provenance: Literal["source", "generated_irrelevant", "legacy_unverified"] = (
        "legacy_unverified"
    )
    question_type: QuestionType
    gold_atom_ids: list[str] = Field(default_factory=list)
    supporting_atom_ids: list[str] = Field(default_factory=list)
    distractor_atom_ids: list[str] = Field(default_factory=list)
    relevant_atom_ids: list[str] = Field(default_factory=list)
    is_irrelevant: bool = False
    has_conflict: bool = False
    is_generated_irrelevant: bool = False
    refusal_target: str | None = None
    refusal_target_derived: bool = False
    annotation_warnings: list[str] = Field(default_factory=list)

    @field_validator("question_id")
    @classmethod
    def require_question_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question_id must not be empty")
        return value

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_answer(cls, value: Any) -> Any:
        if not isinstance(value, dict) or "answers" in value or "answer" not in value:
            return value
        migrated = dict(value)
        legacy = migrated.pop("answer")
        if legacy is None:
            migrated["answers"] = []
        elif isinstance(legacy, list):
            migrated["answers"] = legacy
        else:
            migrated["answers"] = [legacy]
        migrated["answer_provenance"] = "legacy_unverified"
        return migrated

    @property
    def answer(self) -> str | list[str] | None:
        """Read-only compatibility view for schema-v1 consumers."""
        if not self.answers:
            return None
        if len(self.answers) == 1:
            return self.answers[0]
        return list(self.answers)


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
    transformations: list[TransformationEvent] = Field(default_factory=list)
    prompt_provenance: str = "RECONSTRUCTED_PROMPT_NOT_OFFICIAL"


class StructuredRecord(BaseModel):
    schema_version: Literal[2] = ATOMIZATION_SCHEMA_VERSION
    record_id: str
    dataset: str | None = None
    split: str | None = None
    context: str
    atoms: list[Atom]
    qa_pairs: list[QuestionAnnotation] = Field(default_factory=list)
    annotation_meta: AnnotationMeta = Field(default_factory=AnnotationMeta)

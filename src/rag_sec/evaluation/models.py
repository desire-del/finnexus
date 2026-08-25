from typing import Any, Literal

from pydantic import Field

from rag_sec.schemas.base import FinNexusSchema


class ReferenceEvidence(FinNexusSchema):
    text: str | None = None

    document_id: str | None = None
    accession_number: str | None = None
    chunk_id: str | None = None

    section: str | None = None
    item: str | None = None
    page: int | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationCase(FinNexusSchema):
    id: str

    dataset_name: str
    dataset_version: str | None = None

    question: str

    ticker: str | None = None
    company: str | None = None
    form_type: str | None = None
    accession_number: str | None = None

    reference_answer: str | None = None
    reference_evidence: list[ReferenceEvidence] = Field(default_factory=list)

    answerable: bool | None = None

    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedEvidence(FinNexusSchema):
    text: str

    rank: int

    chunk_id: str | None = None
    accession_number: str | None = None

    section: str | None = None
    item: str | None = None
    page: int | None = None

    score: float | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationRunMetrics(FinNexusSchema):
    total_latency_ms: float | None = None
    embedding_latency_ms: float | None = None
    retrieval_latency_ms: float | None = None
    generation_latency_ms: float | None = None

    input_tokens: int | None = None
    output_tokens: int | None = None

    estimated_cost: float | None = None


class EvaluationRun(FinNexusSchema):
    case_id: str

    generated_answer: str | None = None
    retrieved_evidence: list[RetrievedEvidence] = Field(default_factory=list)
    cited_source_ids: list[str] = Field(default_factory=list)

    abstained: bool | None = None

    metrics: EvaluationRunMetrics = Field(default_factory=EvaluationRunMetrics)

    error: str | None = None


class EvaluationScore(FinNexusSchema):
    evaluator: str
    metric: str

    score: float | None = None
    label: str | None = None
    explanation: str | None = None

    evaluator_type: Literal[
        "deterministic",
        "llm_judge",
        "human",
    ]

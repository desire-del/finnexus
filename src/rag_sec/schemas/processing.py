from datetime import datetime
from uuid import UUID

from pydantic import Field

from rag_sec.schemas.base import FinNexusSchema
from rag_sec.schemas.enums import (
    DistanceMetric,
    ProcessingStatus,
)


class ProcessingVersionCreate(FinNexusSchema):
    filing_id: UUID

    ingestion_run_id: UUID

    pipeline_version: str = Field(
        min_length=1,
    )

    parser_name: str = Field(
        min_length=1,
    )

    parser_version: str = Field(
        min_length=1,
    )

    normalization_version: str = Field(
        min_length=1,
    )

    chunking_strategy: str = Field(
        min_length=1,
    )

    chunking_version: str = Field(
        min_length=1,
    )

    embedding_provider: str = Field(
        min_length=1,
    )

    embedding_model: str = Field(
        min_length=1,
    )

    embedding_revision: str | None = None

    embedding_dimension: int = Field(
        gt=0,
    )

    embedding_normalized: bool | None = None

    embedding_instruction: str | None = None

    distance_metric: DistanceMetric = DistanceMetric.COSINE

    processing_fingerprint: str = Field(
        min_length=1,
        description=(
            "Deterministic fingerprint representing the source "
            "and processing configuration."
        ),
    )

    index_version: str | None = None


class ProcessingVersionRead(ProcessingVersionCreate):
    id: UUID

    status: ProcessingStatus

    chunk_count: int = 0

    created_at: datetime

    activated_at: datetime | None = None

    completed_at: datetime | None = None
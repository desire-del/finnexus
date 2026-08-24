from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import Field, HttpUrl

from rag_sec.schemas.base import FinNexusSchema


class ChunkLocator(FinNexusSchema):
    """
    Precise location of a chunk inside the original SEC filing.
    """

    section: str | None = None

    part: str | None = None

    item: str | None = None

    page: int | None = Field(
        default=None,
        ge=1,
    )

    start_char: int | None = Field(
        default=None,
        ge=0,
    )

    end_char: int | None = Field(
        default=None,
        ge=0,
    )

    source_url: HttpUrl | None = None


class ChunkBase(FinNexusSchema):
    chunk_id: str = Field(
        min_length=1,
        description="Stable chunk identifier.",
    )

    filing_id: UUID

    processing_version_id: UUID

    chunk_index: int = Field(
        ge=0,
    )

    text: str = Field(
        min_length=1,
    )

    content_hash: str = Field(
        min_length=1,
    )

    locator: ChunkLocator

    heading_path: list[str] = Field(
        default_factory=list,
    )

    token_count: int | None = Field(
        default=None,
        ge=0,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Non-critical additional metadata.",
    )


class ChunkDraft(ChunkBase):
    """
    Chunk produced after splitting, before embedding.
    """


class EmbeddedChunk(ChunkBase):
    """
    Chunk after embedding generation.
    """

    embedding: list[float] = Field(
        min_length=1,
    )


class ChunkRead(EmbeddedChunk):
    id: UUID

    created_at: datetime

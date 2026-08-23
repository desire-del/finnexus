from typing import TYPE_CHECKING, Any
from uuid import UUID

from pgvector.sqlalchemy import VECTOR
from sqlalchemy import (
    ForeignKeyConstraint,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from rag_sec.models.base import (
    Base,
    TimestampMixin,
    UUIDMixin,
)

if TYPE_CHECKING:
    from rag_sec.models.processing_version import ProcessingVersion


class Chunk(
    Base,
    UUIDMixin,
    TimestampMixin,
):
    __tablename__ = "chunks"

    __table_args__ = (
        ForeignKeyConstraint(
            [
                "processing_version_id",
                "filing_id",
            ],
            [
                "processing_versions.id",
                "processing_versions.filing_id",
            ],
            ondelete="CASCADE",
        ),
        UniqueConstraint(
            "processing_version_id",
            "chunk_index",
            name="uq_chunk_processing_version_index",
        ),
    )

    chunk_id: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
    )

    filing_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=False,
        index=True,
    )

    processing_version_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=False,
        index=True,
    )

    chunk_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    content_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
    )

    # --------------------------------------
    # Provenance / locator
    # --------------------------------------

    section: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    part: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
        index=True,
    )

    item: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )

    page: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    start_char: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    end_char: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    source_url: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    heading_path: Mapped[list[str]] = mapped_column(
        JSONB,
        default=list,
        nullable=False,
    )

    token_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )

    # Vector
    embedding: Mapped[list[float] | None] = mapped_column(
        VECTOR(),
        nullable=True,
    )

    processing_version: Mapped["ProcessingVersion"] = relationship(
        back_populates="chunks",
    )

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from rag_sec.models.base import (
    Base,
    TimestampMixin,
    UUIDMixin,
)

if TYPE_CHECKING:
    from rag_sec.models.filing import Filing
    from rag_sec.models.chunk import Chunk


class ProcessingVersion(
    Base,
    UUIDMixin,
    TimestampMixin,
):
    __tablename__ = "processing_versions"

    __table_args__ = (
        UniqueConstraint(
            "filing_id",
            "processing_fingerprint",
            name="uq_processing_version_filing_fingerprint",
        ),
        UniqueConstraint(
            "id",
            "filing_id",
            name="uq_processing_version_id_filing",
        ),
    )

    filing_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "filings.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    ingestion_run_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "ingestion_runs.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    pipeline_version: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    parser_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    parser_version: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    normalization_version: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    chunking_strategy: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    chunking_version: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    embedding_provider: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    embedding_model: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    embedding_revision: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    embedding_dimension: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    embedding_normalized: Mapped[bool | None] = mapped_column(
        Boolean,
        nullable=True,
    )

    embedding_instruction: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
    )

    distance_metric: Mapped[str] = mapped_column(
        String(30),
        default="cosine",
        nullable=False,
    )

    processing_fingerprint: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    index_version: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    status: Mapped[str] = mapped_column(
        String(50),
        default="building",
        nullable=False,
        index=True,
    )

    chunk_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    activated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    filing: Mapped["Filing"] = relationship(
        back_populates="processing_versions",
    )

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="processing_version",
        cascade="all, delete-orphan",
    )
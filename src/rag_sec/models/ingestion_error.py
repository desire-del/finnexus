from datetime import datetime
from uuid import UUID

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from rag_sec.models.base import Base, UUIDMixin


class IngestionError(
    Base,
    UUIDMixin,
):
    __tablename__ = "ingestion_errors"

    ingestion_run_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "ingestion_runs.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    filing_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "filings.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    accession_number: Mapped[str | None] = mapped_column(
        String(30),
        nullable=True,
    )

    stage: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
    )

    error_type: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    message: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    retriable: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
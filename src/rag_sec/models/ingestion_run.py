from datetime import datetime

from sqlalchemy import DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from rag_sec.models.base import Base, UUIDMixin


class IngestionRun(
    Base,
    UUIDMixin,
):
    __tablename__ = "ingestion_runs"

    pipeline_version: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
    )

    current_stage: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )

    filings_discovered: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    filings_processed: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    filings_skipped: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    filings_failed: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

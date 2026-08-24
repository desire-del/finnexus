from datetime import date, datetime
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import (
    BigInteger,
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from rag_sec.models.base import (
    Base,
    TimestampMixin,
    UUIDMixin,
)

if TYPE_CHECKING:
    from rag_sec.models.company import Company
    from rag_sec.models.processing_version import ProcessingVersion


class Filing(
    Base,
    UUIDMixin,
    TimestampMixin,
):
    __tablename__ = "filings"

    company_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "companies.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    accession_number: Mapped[str] = mapped_column(
        String(30),
        unique=True,
        nullable=False,
        index=True,
    )

    form_type: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        index=True,
    )

    filing_date: Mapped[date] = mapped_column(
        Date,
        nullable=False,
        index=True,
    )

    period_of_report: Mapped[date | None] = mapped_column(
        Date,
        nullable=True,
        index=True,
    )

    acceptance_datetime: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    file_number: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )

    primary_document: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    primary_document_description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    source_uri: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    filing_url: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    homepage_url: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    is_xbrl: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    is_inline_xbrl: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    is_amendment: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    authority: Mapped[str] = mapped_column(
        String(50),
        default="sec_official",
        nullable=False,
    )

    content_hash: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        index=True,
    )

    source_size_bytes: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True,
    )

    status: Mapped[str] = mapped_column(
        String(50),
        default="discovered",
        nullable=False,
        index=True,
    )

    discovered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    fetched_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    company: Mapped["Company"] = relationship(
        back_populates="filings",
    )

    processing_versions: Mapped[list["ProcessingVersion"]] = relationship(
        back_populates="filing",
        cascade="all, delete-orphan",
    )

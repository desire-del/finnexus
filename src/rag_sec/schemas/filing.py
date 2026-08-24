from datetime import date, datetime
from uuid import UUID

from pydantic import Field, HttpUrl

from rag_sec.schemas.base import FinNexusSchema
from rag_sec.schemas.enums import (
    FilingStatus,
    SourceAuthority,
)


class FilingBase(FinNexusSchema):
    company_id: UUID

    accession_number: str = Field(
        min_length=1,
        description="Unique SEC accession number.",
    )

    form_type: str = Field(
        min_length=1,
        description="SEC form type, e.g. 10-K, 10-Q, 8-K.",
    )

    filing_date: date

    period_of_report: date | None = None

    acceptance_datetime: datetime | None = None

    file_number: str | None = None

    primary_document: str | None = None

    primary_document_description: str | None = None

    source_uri: HttpUrl = Field(
        description=(
            "URI of the original SEC artifact. "
            "Prefer the full filing/text submission URL."
        )
    )

    filing_url: HttpUrl | None = None

    homepage_url: HttpUrl | None = None

    is_xbrl: bool = False

    is_inline_xbrl: bool = False

    is_amendment: bool = False

    authority: SourceAuthority = SourceAuthority.SEC_OFFICIAL


class FilingCreate(FilingBase):
    """
    Metadata available when a filing is first discovered
    through EdgarTools.
    """


class FilingRead(FilingBase):
    """
    Persisted filing registry entry.
    """

    id: UUID

    content_hash: str | None = None

    source_size_bytes: int | None = None

    status: FilingStatus

    discovered_at: datetime

    fetched_at: datetime | None = None

    created_at: datetime

    updated_at: datetime


class FilingContent(FinNexusSchema):
    """
    Raw content after fetching the SEC filing.

    This object flows through the ingestion pipeline;
    it is not necessarily stored as-is in PostgreSQL.
    """

    filing_id: UUID

    accession_number: str

    content: str = Field(
        min_length=1,
    )

    content_hash: str = Field(
        min_length=1,
    )

    source_size_bytes: int = Field(
        ge=0,
    )

    source_uri: HttpUrl

    fetched_at: datetime


class FilingSection(FinNexusSchema):
    name: str

    part: str | None = None

    item: str | None = None

    content: str = Field(
        min_length=1,
    )

    warnings: list[str] = Field(
        default_factory=list,
    )

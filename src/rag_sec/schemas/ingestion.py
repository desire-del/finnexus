from datetime import date, datetime
from uuid import UUID

from pydantic import Field

from rag_sec.schemas.base import FinNexusSchema
from rag_sec.schemas.enums import (
    IngestionStage,
    IngestionStatus,
)


class IngestionRequest(FinNexusSchema):
    """
    Request describing what should be fetched from SEC EDGAR.
    """

    company_identifiers: list[str] = Field(
        min_length=1,
        description=(
            "Tickers or CIK values used to identify companies."
        ),
    )

    form_types: list[str] = Field(
        default_factory=lambda: ["10-K", "10-Q"],
    )

    start_date: date | None = None

    end_date: date | None = None

    include_amendments: bool = False

    limit_per_company: int | None = Field(
        default=None,
        gt=0,
    )


class IngestionRunCreate(FinNexusSchema):
    pipeline_version: str = Field(
        min_length=1,
    )


class IngestionRunRead(FinNexusSchema):
    id: UUID

    pipeline_version: str

    status: IngestionStatus

    current_stage: IngestionStage | None = None

    filings_discovered: int = 0

    filings_processed: int = 0

    filings_skipped: int = 0

    filings_failed: int = 0

    started_at: datetime

    completed_at: datetime | None = None


class IngestionError(FinNexusSchema):
    ingestion_run_id: UUID

    filing_id: UUID | None = None

    accession_number: str | None = None

    stage: IngestionStage

    error_type: str

    message: str

    retriable: bool = False

    occurred_at: datetime


class IngestionResult(FinNexusSchema):
    ingestion_run_id: UUID

    status: IngestionStatus

    filings_discovered: int

    filings_processed: int

    filings_skipped: int

    filings_failed: int

    processed_filing_ids: list[UUID] = Field(
        default_factory=list,
    )

    failed_accession_numbers: list[str] = Field(
        default_factory=list,
    )
from datetime import date
from uuid import UUID

from pydantic import BaseModel
from sqlalchemy import func, select

from rag_sec.application.runtime import RAGRuntime
from rag_sec.config import get_settings
from rag_sec.models.company import Company
from rag_sec.models.filing import Filing
from rag_sec.models.processing_version import ProcessingVersion
from rag_sec.schemas.enums import ProcessingStatus


class AvailableFiling(BaseModel):
    """An active filing that can be queried by the configured retriever."""

    filing_id: UUID
    company_name: str
    ticker: str | None
    cik: int
    accession_number: str
    form_type: str
    filing_date: date
    period_of_report: date | None
    source_url: str | None
    chunk_count: int
    embedding_provider: str
    embedding_model: str
    embedding_dimension: int


async def list_available_filings(
    runtime: RAGRuntime,
    *,
    limit: int = 200,
) -> list[AvailableFiling]:
    """List active filings compatible with the current embedding profile."""
    if not 1 <= limit <= 500:
        raise ValueError("The filing limit must be between 1 and 500.")

    embedding = get_settings().embedding
    statement = (
        select(
            Filing.id.label("filing_id"),
            Company.name.label("company_name"),
            Company.ticker,
            Company.cik,
            Filing.accession_number,
            Filing.form_type,
            Filing.filing_date,
            Filing.period_of_report,
            func.coalesce(
                Filing.filing_url,
                Filing.source_uri,
            ).label("source_url"),
            ProcessingVersion.chunk_count,
            ProcessingVersion.embedding_provider,
            ProcessingVersion.embedding_model,
            ProcessingVersion.embedding_dimension,
        )
        .join(
            Company,
            Filing.company_id == Company.id,
        )
        .join(
            ProcessingVersion,
            ProcessingVersion.filing_id == Filing.id,
        )
        .where(
            ProcessingVersion.status == ProcessingStatus.ACTIVE.value,
            ProcessingVersion.embedding_provider == embedding.provider.value,
            ProcessingVersion.embedding_model == embedding.model_name,
            ProcessingVersion.embedding_dimension == embedding.dimension,
        )
        .order_by(Filing.filing_date.desc(), Company.ticker.asc())
        .limit(limit)
    )

    async with runtime.database.session() as session:
        result = await session.execute(statement)
        rows = result.mappings().all()

    return [AvailableFiling.model_validate(row) for row in rows]

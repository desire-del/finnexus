from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_sec.models.filing import Filing
from rag_sec.schemas.enums import FilingStatus
from rag_sec.schemas.filing import FilingCreate


class FilingRepository:
    @staticmethod
    async def get_by_id(session: AsyncSession, filing_id: UUID) -> Filing | None:
        return await session.get(Filing, filing_id)

    @staticmethod
    async def get_by_accession_number(
        session: AsyncSession, accession_number: str
    ) -> Filing | None:
        result = await session.execute(
            select(Filing).where(Filing.accession_number == accession_number)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def create(session: AsyncSession, data: FilingCreate) -> Filing:
        filing = Filing(
            company_id=data.company_id,
            accession_number=data.accession_number,
            form_type=data.form_type,
            filing_date=data.filing_date,
            period_of_report=data.period_of_report,
            acceptance_datetime=data.acceptance_datetime,
            file_number=data.file_number,
            primary_document=data.primary_document,
            primary_document_description=data.primary_document_description,
            source_uri=str(data.source_uri),
            filing_url=str(data.filing_url) if data.filing_url else None,
            homepage_url=str(data.homepage_url) if data.homepage_url else None,
            is_xbrl=data.is_xbrl,
            is_inline_xbrl=data.is_inline_xbrl,
            is_amendment=data.is_amendment,
            authority=data.authority.value,
            status=FilingStatus.DISCOVERED.value,
            discovered_at=datetime.now(UTC),
        )
        session.add(filing)
        await session.flush()
        await session.refresh(filing)
        return filing

    @staticmethod
    async def mark_fetched(
        session: AsyncSession,
        filing: Filing,
        *,
        content_hash: str,
        source_size_bytes: int,
    ) -> Filing:
        filing.content_hash = content_hash
        filing.source_size_bytes = source_size_bytes
        filing.status = FilingStatus.FETCHED.value
        filing.fetched_at = datetime.now(UTC)
        await session.flush()
        return filing

    @staticmethod
    async def mark_failed(session: AsyncSession, filing: Filing) -> Filing:
        filing.status = FilingStatus.FAILED.value
        await session.flush()
        return filing

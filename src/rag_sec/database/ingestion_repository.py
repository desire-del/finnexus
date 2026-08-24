from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from rag_sec.models.ingestion_error import IngestionError as IngestionErrorModel
from rag_sec.models.ingestion_run import IngestionRun
from rag_sec.schemas.enums import IngestionStatus
from rag_sec.schemas.ingestion import IngestionError, IngestionRunCreate


class IngestionRepository:
    @staticmethod
    async def get_by_id(session: AsyncSession, run_id: UUID) -> IngestionRun | None:
        return await session.get(IngestionRun, run_id)

    @staticmethod
    async def create_run(
        session: AsyncSession, data: IngestionRunCreate
    ) -> IngestionRun:
        run = IngestionRun(
            pipeline_version=data.pipeline_version,
            status=IngestionStatus.RUNNING.value,
            current_stage=None,
            filings_discovered=0,
            filings_processed=0,
            filings_skipped=0,
            filings_failed=0,
            started_at=datetime.now(UTC),
        )
        session.add(run)
        await session.flush()
        await session.refresh(run)
        return run

    @staticmethod
    async def set_stage(session: AsyncSession, run: IngestionRun, stage: str) -> None:
        run.current_stage = stage
        await session.flush()

    @staticmethod
    async def increment_discovered(
        session: AsyncSession, run: IngestionRun, amount: int = 1
    ) -> None:
        run.filings_discovered += amount
        await session.flush()

    @staticmethod
    async def increment_processed(
        session: AsyncSession, run: IngestionRun, amount: int = 1
    ) -> None:
        run.filings_processed += amount
        await session.flush()

    @staticmethod
    async def increment_skipped(
        session: AsyncSession, run: IngestionRun, amount: int = 1
    ) -> None:
        run.filings_skipped += amount
        await session.flush()

    @staticmethod
    async def increment_failed(
        session: AsyncSession, run: IngestionRun, amount: int = 1
    ) -> None:
        run.filings_failed += amount
        await session.flush()

    @staticmethod
    async def add_error(
        session: AsyncSession, data: IngestionError
    ) -> IngestionErrorModel:
        error = IngestionErrorModel(
            ingestion_run_id=data.ingestion_run_id,
            filing_id=data.filing_id,
            accession_number=data.accession_number,
            stage=data.stage.value,
            error_type=data.error_type,
            message=data.message,
            retriable=data.retriable,
            occurred_at=data.occurred_at,
        )
        session.add(error)
        await session.flush()
        await session.refresh(error)
        return error

    @staticmethod
    async def complete_run(session: AsyncSession, run: IngestionRun) -> IngestionRun:
        run.status = (
            IngestionStatus.COMPLETED_WITH_ERRORS.value
            if run.filings_failed
            else IngestionStatus.COMPLETED.value
        )
        run.current_stage = None
        run.completed_at = datetime.now(UTC)
        await session.flush()
        return run

    @staticmethod
    async def fail_run(session: AsyncSession, run: IngestionRun) -> IngestionRun:
        run.status = IngestionStatus.FAILED.value
        run.current_stage = None
        run.completed_at = datetime.now(UTC)
        await session.flush()
        return run

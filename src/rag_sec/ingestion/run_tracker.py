from datetime import UTC, datetime
from uuid import UUID

from rag_sec.database.ingestion_repository import IngestionRepository
from rag_sec.database.manager import DatabaseManager
from rag_sec.database.processing_repository import ProcessingRepository
from rag_sec.logging import get_logger
from rag_sec.observability import set_span_attributes
from rag_sec.schemas.enums import IngestionStage, IngestionStatus
from rag_sec.schemas.ingestion import (
    IngestionError,
    IngestionResult,
    IngestionRunCreate,
)

log = get_logger(__name__)


class IngestionRunTracker:
    """Persist the lifecycle and failures of ingestion attempts."""

    def __init__(self, database: DatabaseManager, *, pipeline_version: str) -> None:
        self.database = database
        self.pipeline_version = pipeline_version

    async def create(self) -> UUID:
        async with self.database.session() as session:
            run = await IngestionRepository.create_run(
                session, IngestionRunCreate(pipeline_version=self.pipeline_version)
            )
            return run.id

    async def set_stage(self, run_id: UUID, stage: IngestionStage) -> None:
        set_span_attributes({"rag.ingestion.stage": stage.value})
        async with self.database.session() as session:
            run = await IngestionRepository.get_by_id(session, run_id)
            if run is None:
                raise RuntimeError("Ingestion run not found.")
            await IngestionRepository.set_stage(session, run, stage.value)

    async def complete_skipped(
        self, run_id: UUID, *, accession_number: str
    ) -> IngestionResult:
        async with self.database.session() as session:
            run = await IngestionRepository.get_by_id(session, run_id)
            if run is None:
                raise RuntimeError("Ingestion run not found.")
            await IngestionRepository.increment_skipped(session, run)
            await IngestionRepository.complete_run(session, run)
        log.info(
            "filing_ingestion_skipped",
            accession_number=accession_number,
            reason="processing_version_already_active",
        )
        return IngestionResult(
            ingestion_run_id=run_id,
            status=IngestionStatus.COMPLETED,
            filings_discovered=1,
            filings_processed=0,
            filings_skipped=1,
            filings_failed=0,
        )

    async def record_failure(
        self,
        *,
        run_id: UUID,
        filing_id: UUID | None,
        processing_version_id: UUID | None,
        accession_number: str | None,
        stage: IngestionStage,
        error: Exception,
    ) -> None:
        """Persist failure state without hiding the original exception."""
        try:
            async with self.database.session() as session:
                run = await IngestionRepository.get_by_id(session, run_id)
                if run is not None:
                    await IngestionRepository.increment_failed(session, run)
                    await IngestionRepository.add_error(
                        session,
                        IngestionError(
                            ingestion_run_id=run_id,
                            filing_id=filing_id,
                            accession_number=accession_number,
                            stage=stage,
                            error_type=type(error).__name__,
                            message=str(error),
                            retriable=False,
                            occurred_at=datetime.now(UTC),
                        ),
                    )
                    await IngestionRepository.fail_run(session, run)
                if processing_version_id is not None:
                    version = await ProcessingRepository.get_by_id(
                        session, processing_version_id
                    )
                    if version is not None:
                        await ProcessingRepository.mark_failed(session, version)
        except Exception as persistence_error:  # noqa: BLE001
            log.error(
                "failed_to_record_ingestion_failure",
                original_error=str(error),
                persistence_error=str(persistence_error),
            )

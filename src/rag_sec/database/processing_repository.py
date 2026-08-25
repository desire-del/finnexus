from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from rag_sec.models.chunk import Chunk
from rag_sec.models.company import Company
from rag_sec.models.filing import Filing
from rag_sec.models.processing_version import ProcessingVersion
from rag_sec.schemas.chunk import EmbeddedChunk
from rag_sec.schemas.enums import ProcessingStatus
from rag_sec.schemas.processing import ProcessingVersionCreate


class ProcessingRepository:
    @staticmethod
    async def get_by_id(
        session: AsyncSession, processing_version_id: UUID
    ) -> ProcessingVersion | None:
        return await session.get(ProcessingVersion, processing_version_id)

    @staticmethod
    async def get_by_fingerprint(
        session: AsyncSession,
        *,
        filing_id: UUID,
        processing_fingerprint: str,
    ) -> ProcessingVersion | None:
        result = await session.execute(
            select(ProcessingVersion).where(
                ProcessingVersion.filing_id == filing_id,
                ProcessingVersion.processing_fingerprint == processing_fingerprint,
            )
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_active_version(
        session: AsyncSession,
        *,
        filing_id: UUID,
        embedding_provider: str,
        embedding_model: str,
        embedding_dimension: int,
    ) -> ProcessingVersion | None:
        result = await session.execute(
            select(ProcessingVersion).where(
                ProcessingVersion.filing_id == filing_id,
                ProcessingVersion.status == ProcessingStatus.ACTIVE.value,
                ProcessingVersion.embedding_provider == embedding_provider,
                ProcessingVersion.embedding_model == embedding_model,
                ProcessingVersion.embedding_dimension == embedding_dimension,
            )
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def create_version(
        session: AsyncSession, data: ProcessingVersionCreate
    ) -> ProcessingVersion:
        version = ProcessingVersion(
            **data.model_dump(exclude={"distance_metric"}),
            distance_metric=data.distance_metric.value,
            status=ProcessingStatus.BUILDING.value,
        )
        session.add(version)
        await session.flush()
        await session.refresh(version)
        return version

    @staticmethod
    async def add_chunks(
        session: AsyncSession, chunks: list[EmbeddedChunk]
    ) -> list[Chunk]:
        models = [ProcessingRepository._chunk_model(chunk) for chunk in chunks]
        session.add_all(models)
        await session.flush()
        return models

    @staticmethod
    def _chunk_model(chunk: EmbeddedChunk) -> Chunk:
        return Chunk(
            chunk_id=chunk.chunk_id,
            filing_id=chunk.filing_id,
            processing_version_id=chunk.processing_version_id,
            chunk_index=chunk.chunk_index,
            text=chunk.text,
            content_hash=chunk.content_hash,
            section=chunk.locator.section,
            part=chunk.locator.part,
            item=chunk.locator.item,
            page=chunk.locator.page,
            start_char=chunk.locator.start_char,
            end_char=chunk.locator.end_char,
            source_url=(
                str(chunk.locator.source_url) if chunk.locator.source_url else None
            ),
            heading_path=chunk.heading_path,
            token_count=chunk.token_count,
            metadata_=chunk.metadata,
            embedding=chunk.embedding,
        )

    @staticmethod
    async def count_chunks(session: AsyncSession, processing_version_id: UUID) -> int:
        result = await session.execute(
            select(func.count(Chunk.id)).where(
                Chunk.processing_version_id == processing_version_id
            )
        )
        return result.scalar_one()

    @staticmethod
    async def mark_failed(
        session: AsyncSession, version: ProcessingVersion
    ) -> ProcessingVersion:
        version.status = ProcessingStatus.FAILED.value
        version.completed_at = datetime.now(UTC)
        await session.flush()
        return version

    @staticmethod
    async def reset_version(
        session: AsyncSession,
        processing_version_id: UUID,
        ingestion_run_id: UUID,
    ) -> ProcessingVersion:
        """Atomically remove partial chunks and reuse a retryable version."""
        result = await session.execute(
            select(ProcessingVersion)
            .where(ProcessingVersion.id == processing_version_id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        version = result.scalar_one_or_none()
        if version is None:
            raise ValueError(
                f"Processing version not found for reset: {processing_version_id}."
            )
        if version.status not in {
            ProcessingStatus.FAILED.value,
            ProcessingStatus.BUILDING.value,
        }:
            raise ValueError(
                f"Cannot reset processing version {processing_version_id} "
                f"with status '{version.status}'."
            )

        await session.execute(
            delete(Chunk).where(Chunk.processing_version_id == version.id)
        )
        version.ingestion_run_id = ingestion_run_id
        version.status = ProcessingStatus.BUILDING.value
        version.chunk_count = 0
        version.activated_at = None
        version.completed_at = None
        await session.flush()
        return version

    @staticmethod
    async def activate(
        session: AsyncSession, version: ProcessingVersion
    ) -> ProcessingVersion:
        chunk_count = await ProcessingRepository.count_chunks(session, version.id)
        if chunk_count == 0:
            raise ValueError("Cannot activate a processing version without chunks.")

        result = await session.execute(
            select(func.count(Chunk.id)).where(
                Chunk.processing_version_id == version.id,
                Chunk.embedding.is_(None),
            )
        )
        missing_embeddings = result.scalar_one()
        if missing_embeddings:
            raise ValueError(
                "Cannot activate processing version: "
                f"{missing_embeddings} chunks do not have embeddings."
            )

        await session.execute(
            update(ProcessingVersion)
            .where(
                ProcessingVersion.filing_id == version.filing_id,
                ProcessingVersion.id != version.id,
                ProcessingVersion.status == ProcessingStatus.ACTIVE.value,
                ProcessingVersion.embedding_provider == version.embedding_provider,
                ProcessingVersion.embedding_model == version.embedding_model,
                ProcessingVersion.embedding_dimension == version.embedding_dimension,
            )
            .values(status=ProcessingStatus.SUPERSEDED.value)
        )
        now = datetime.now(UTC)
        version.status = ProcessingStatus.ACTIVE.value
        version.chunk_count = chunk_count
        version.completed_at = now
        version.activated_at = now
        await session.flush()
        return version

    @staticmethod
    async def get_active_ids(
        session: AsyncSession,
        *,
        ticker: str | None = None,
        form_type: str | None = None,
        accession_number: str | None = None,
        embedding_provider: str | None = None,
        embedding_model: str | None = None,
        embedding_dimension: int | None = None,
    ) -> list[UUID]:
        statement = (
            select(ProcessingVersion.id)
            .join(Filing, ProcessingVersion.filing_id == Filing.id)
            .join(Company, Filing.company_id == Company.id)
            .where(ProcessingVersion.status == ProcessingStatus.ACTIVE.value)
        )
        filters = (
            (ticker, Company.ticker == ticker.upper() if ticker else None),
            (form_type, Filing.form_type == form_type if form_type else None),
            (
                accession_number,
                Filing.accession_number == accession_number
                if accession_number
                else None,
            ),
            (
                embedding_provider,
                ProcessingVersion.embedding_provider == embedding_provider
                if embedding_provider
                else None,
            ),
            (
                embedding_model,
                ProcessingVersion.embedding_model == embedding_model
                if embedding_model
                else None,
            ),
            (
                embedding_dimension,
                ProcessingVersion.embedding_dimension == embedding_dimension
                if embedding_dimension is not None
                else None,
            ),
        )
        for value, criterion in filters:
            if value is not None and criterion is not None:
                statement = statement.where(criterion)
        result = await session.execute(statement)
        return list(result.scalars().all())

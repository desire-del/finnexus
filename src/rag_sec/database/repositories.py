# src/rag_sec/database/repositories.py

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from rag_sec.models.company import Company
from rag_sec.models.filing import Filing
from rag_sec.models.processing_version import ProcessingVersion
from rag_sec.models.chunk import Chunk
from rag_sec.models.ingestion_run import IngestionRun
from rag_sec.models.ingestion_error import IngestionError as IngestionErrorModel

from rag_sec.schemas.company import CompanyCreate
from rag_sec.schemas.filing import FilingCreate
from rag_sec.schemas.processing import ProcessingVersionCreate
from rag_sec.schemas.chunk import EmbeddedChunk
from rag_sec.schemas.ingestion import (
    IngestionRunCreate,
    IngestionError as IngestionErrorSchema,
)

from rag_sec.schemas.enums import (
    FilingStatus,
    ProcessingStatus,
    IngestionStatus,
)


class CompanyRepository:

    @staticmethod
    async def get_by_cik(
        session: AsyncSession,
        cik: int,
    ) -> Company | None:
        result = await session.execute(
            select(Company).where(
                Company.cik == cik
            )
        )

        return result.scalar_one_or_none()

    @staticmethod
    async def get_by_ticker(
        session: AsyncSession,
        ticker: str,
    ) -> Company | None:
        result = await session.execute(
            select(Company).where(
                Company.ticker == ticker.upper()
            )
        )

        return result.scalar_one_or_none()

    @staticmethod
    async def create(
        session: AsyncSession,
        data: CompanyCreate,
    ) -> Company:
        company = Company(
            cik=data.cik,
            name=data.name,
            ticker=data.ticker,
        )

        session.add(company)

        await session.flush()
        await session.refresh(company)

        return company

    @classmethod
    async def get_or_create(
        cls,
        session: AsyncSession,
        data: CompanyCreate,
    ) -> Company:
        company = await cls.get_by_cik(
            session=session,
            cik=data.cik,
        )

        if company is not None:
            return company

        return await cls.create(
            session=session,
            data=data,
        )


class FilingRepository:

    @staticmethod
    async def get_by_id(
        session: AsyncSession,
        filing_id: UUID,
    ) -> Filing | None:
        return await session.get(
            Filing,
            filing_id,
        )

    @staticmethod
    async def get_by_accession_number(
        session: AsyncSession,
        accession_number: str,
    ) -> Filing | None:
        result = await session.execute(
            select(Filing).where(
                Filing.accession_number == accession_number
            )
        )

        return result.scalar_one_or_none()

    @staticmethod
    async def create(
        session: AsyncSession,
        data: FilingCreate,
    ) -> Filing:
        filing = Filing(
            company_id=data.company_id,
            accession_number=data.accession_number,
            form_type=data.form_type,
            filing_date=data.filing_date,
            period_of_report=data.period_of_report,
            acceptance_datetime=data.acceptance_datetime,
            file_number=data.file_number,
            primary_document=data.primary_document,
            primary_document_description=(
                data.primary_document_description
            ),
            source_uri=str(data.source_uri),
            filing_url=(
                str(data.filing_url)
                if data.filing_url
                else None
            ),
            homepage_url=(
                str(data.homepage_url)
                if data.homepage_url
                else None
            ),
            is_xbrl=data.is_xbrl,
            is_inline_xbrl=data.is_inline_xbrl,
            is_amendment=data.is_amendment,
            authority=data.authority.value,
            status=FilingStatus.DISCOVERED.value,
            discovered_at=datetime.now(timezone.utc),
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
        filing.fetched_at = datetime.now(timezone.utc)

        await session.flush()

        return filing

    @staticmethod
    async def mark_failed(
        session: AsyncSession,
        filing: Filing,
    ) -> Filing:
        filing.status = FilingStatus.FAILED.value

        await session.flush()

        return filing


class ProcessingRepository:

    @staticmethod
    async def get_by_id(
        session: AsyncSession,
        processing_version_id: UUID,
    ) -> ProcessingVersion | None:
        return await session.get(
            ProcessingVersion,
            processing_version_id,
        )

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
                ProcessingVersion.processing_fingerprint
                == processing_fingerprint,
            )
        )

        return result.scalar_one_or_none()

    @staticmethod
    async def get_active_version(
        session: AsyncSession,
        filing_id: UUID,
    ) -> ProcessingVersion | None:
        result = await session.execute(
            select(ProcessingVersion).where(
                ProcessingVersion.filing_id == filing_id,
                ProcessingVersion.status
                == ProcessingStatus.ACTIVE.value,
            )
        )

        return result.scalar_one_or_none()

    @staticmethod
    async def create_version(
        session: AsyncSession,
        data: ProcessingVersionCreate,
    ) -> ProcessingVersion:
        version = ProcessingVersion(
            filing_id=data.filing_id,
            ingestion_run_id=data.ingestion_run_id,
            pipeline_version=data.pipeline_version,
            parser_name=data.parser_name,
            parser_version=data.parser_version,
            normalization_version=data.normalization_version,
            chunking_strategy=data.chunking_strategy,
            chunking_version=data.chunking_version,
            embedding_provider=data.embedding_provider,
            embedding_model=data.embedding_model,
            embedding_revision=data.embedding_revision,
            embedding_dimension=data.embedding_dimension,
            embedding_normalized=data.embedding_normalized,
            embedding_instruction=data.embedding_instruction,
            distance_metric=data.distance_metric.value,
            processing_fingerprint=data.processing_fingerprint,
            index_version=data.index_version,
            status=ProcessingStatus.BUILDING.value,
        )

        session.add(version)

        await session.flush()
        await session.refresh(version)

        return version

    @staticmethod
    async def add_chunks(
        session: AsyncSession,
        chunks: list[EmbeddedChunk],
    ) -> list[Chunk]:
        models: list[Chunk] = []

        for chunk in chunks:
            model = Chunk(
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
                    str(chunk.locator.source_url)
                    if chunk.locator.source_url
                    else None
                ),
                heading_path=chunk.heading_path,
                token_count=chunk.token_count,
                metadata_=chunk.metadata,
                embedding=chunk.embedding,
            )

            models.append(model)

        session.add_all(models)

        await session.flush()

        return models

    @staticmethod
    async def count_chunks(
        session: AsyncSession,
        processing_version_id: UUID,
    ) -> int:
        result = await session.execute(
            select(
                func.count(Chunk.id)
            ).where(
                Chunk.processing_version_id
                == processing_version_id
            )
        )

        return result.scalar_one()

    @staticmethod
    async def mark_failed(
        session: AsyncSession,
        version: ProcessingVersion,
    ) -> ProcessingVersion:
        version.status = ProcessingStatus.FAILED.value
        version.completed_at = datetime.now(timezone.utc)

        await session.flush()

        return version

    @staticmethod
    async def activate(
        session: AsyncSession,
        version: ProcessingVersion,
    ) -> ProcessingVersion:
        result = await session.execute(
            select(
                func.count(Chunk.id)
            ).where(
                Chunk.processing_version_id == version.id
            )
        )

        chunk_count = result.scalar_one()

        if chunk_count == 0:
            raise ValueError(
                "Cannot activate a processing version "
                "without chunks."
            )

        result = await session.execute(
            select(
                func.count(Chunk.id)
            ).where(
                Chunk.processing_version_id == version.id,
                Chunk.embedding.is_(None),
            )
        )

        missing_embeddings = result.scalar_one()

        if missing_embeddings > 0:
            raise ValueError(
                f"Cannot activate processing version: "
                f"{missing_embeddings} chunks "
                f"do not have embeddings."
            )

        await session.execute(
            update(ProcessingVersion)
            .where(
                ProcessingVersion.filing_id
                == version.filing_id,
                ProcessingVersion.id != version.id,
                ProcessingVersion.status
                == ProcessingStatus.ACTIVE.value,
            )
            .values(
                status=ProcessingStatus.SUPERSEDED.value
            )
        )

        now = datetime.now(timezone.utc)

        version.status = ProcessingStatus.ACTIVE.value
        version.chunk_count = chunk_count
        version.completed_at = now
        version.activated_at = now

        await session.flush()

        return version


class IngestionRepository:

    @staticmethod
    async def get_by_id(
        session: AsyncSession,
        run_id: UUID,
    ) -> IngestionRun | None:
        return await session.get(
            IngestionRun,
            run_id,
        )

    @staticmethod
    async def create_run(
        session: AsyncSession,
        data: IngestionRunCreate,
    ) -> IngestionRun:
        run = IngestionRun(
            pipeline_version=data.pipeline_version,
            status=IngestionStatus.RUNNING.value,
            current_stage=None,
            filings_discovered=0,
            filings_processed=0,
            filings_skipped=0,
            filings_failed=0,
            started_at=datetime.now(timezone.utc),
        )

        session.add(run)

        await session.flush()
        await session.refresh(run)

        return run

    @staticmethod
    async def set_stage(
        session: AsyncSession,
        run: IngestionRun,
        stage: str,
    ) -> None:
        run.current_stage = stage

        await session.flush()

    @staticmethod
    async def increment_discovered(
        session: AsyncSession,
        run: IngestionRun,
        amount: int = 1,
    ) -> None:
        run.filings_discovered += amount

        await session.flush()

    @staticmethod
    async def increment_processed(
        session: AsyncSession,
        run: IngestionRun,
        amount: int = 1,
    ) -> None:
        run.filings_processed += amount

        await session.flush()

    @staticmethod
    async def increment_skipped(
        session: AsyncSession,
        run: IngestionRun,
        amount: int = 1,
    ) -> None:
        run.filings_skipped += amount

        await session.flush()

    @staticmethod
    async def increment_failed(
        session: AsyncSession,
        run: IngestionRun,
        amount: int = 1,
    ) -> None:
        run.filings_failed += amount

        await session.flush()

    @staticmethod
    async def add_error(
        session: AsyncSession,
        data: IngestionErrorSchema,
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
    async def complete_run(
        session: AsyncSession,
        run: IngestionRun,
    ) -> IngestionRun:
        if run.filings_failed > 0:
            run.status = (
                IngestionStatus.COMPLETED_WITH_ERRORS.value
            )
        else:
            run.status = IngestionStatus.COMPLETED.value

        run.current_stage = None
        run.completed_at = datetime.now(timezone.utc)

        await session.flush()

        return run

    @staticmethod
    async def fail_run(
        session: AsyncSession,
        run: IngestionRun,
    ) -> IngestionRun:
        run.status = IngestionStatus.FAILED.value
        run.current_stage = None
        run.completed_at = datetime.now(timezone.utc)

        await session.flush()

        return run
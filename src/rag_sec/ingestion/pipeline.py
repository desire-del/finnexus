# src/rag_sec/ingestion/pipeline.py

import hashlib

from datetime import datetime, timezone
from importlib.metadata import version as package_version
from uuid import UUID

from rag_sec.config import get_settings
from rag_sec.database.manager import get_database_manager

from rag_sec.database.repositories import (
    CompanyRepository,
    FilingRepository,
    ProcessingRepository,
    IngestionRepository,
)

from rag_sec.ingestion.edgar_client import EdgarClient
from rag_sec.ingestion.chunker import SectionChunker
from rag_sec.ingestion.embeddings import OpenAIEmbedder

from rag_sec.logging import get_logger

from rag_sec.schemas.processing import (
    ProcessingVersionCreate,
)

from rag_sec.schemas.ingestion import (
    IngestionRunCreate,
    IngestionError,
    IngestionResult,
)

from rag_sec.schemas.enums import (
    DistanceMetric,
    IngestionStage,
    IngestionStatus,
    ProcessingStatus,
)


log = get_logger(__name__)


class IngestionPipeline:
    """
    End-to-end SEC filing ingestion pipeline.
    """

    PIPELINE_VERSION = "ingestion-v1"
    NORMALIZATION_VERSION = "normalization-v1"

    def __init__(self):
        self.settings = get_settings()

        self.db = get_database_manager()

        self.edgar = EdgarClient()

        self.chunker = SectionChunker(
            max_tokens=800,
            overlap_tokens=100,
        )

        self.embedder = OpenAIEmbedder(
            batch_size=64,
        )

        self.parser_version = package_version(
            "edgartools"
        )

    async def ingest_latest(
        self,
        identifier: str | int,
        form_type: str = "10-K",
    ) -> IngestionResult:

        run_id: UUID | None = None
        filing_id: UUID | None = None
        processing_version_id: UUID | None = None

        accession_number: str | None = None

        current_stage = IngestionStage.DISCOVER

        try:
            # --------------------------------------
            # 1. Create ingestion run
            # --------------------------------------

            async with self.db.session() as session:

                run = await IngestionRepository.create_run(
                    session,
                    IngestionRunCreate(
                        pipeline_version=(
                            self.PIPELINE_VERSION
                        )
                    ),
                )

                run_id = run.id

            # --------------------------------------
            # 2. Discover SEC filing
            # --------------------------------------

            await self._set_stage(
                run_id,
                IngestionStage.DISCOVER,
            )

            company, filing = (
                await self.edgar.get_latest_filing(
                    identifier,
                    form_type=form_type,
                )
            )

            accession_number = (
                filing.accession_number
            )

            company_schema = (
                self.edgar.to_company_schema(
                    company
                )
            )

            # --------------------------------------
            # 3. Persist Company + Filing registry
            # --------------------------------------

            async with self.db.session() as session:

                run = await IngestionRepository.get_by_id(
                    session,
                    run_id,
                )

                await IngestionRepository.increment_discovered(
                    session,
                    run,
                )

                company_model = (
                    await CompanyRepository.get_or_create(
                        session,
                        company_schema,
                    )
                )

                filing_model = (
                    await FilingRepository
                    .get_by_accession_number(
                        session,
                        accession_number,
                    )
                )

                if filing_model is None:

                    filing_schema = (
                        self.edgar.to_filing_schema(
                            filing,
                            company_model.id,
                        )
                    )

                    filing_model = (
                        await FilingRepository.create(
                            session,
                            filing_schema,
                        )
                    )

                filing_id = filing_model.id

                existing_content_hash = (
                    filing_model.content_hash
                )

            # --------------------------------------
            # 4. Fast idempotency check
            # --------------------------------------

            if existing_content_hash:

                fingerprint = (
                    self._processing_fingerprint(
                        existing_content_hash
                    )
                )

                async with self.db.session() as session:

                    existing_version = (
                        await ProcessingRepository
                        .get_by_fingerprint(
                            session,
                            filing_id=filing_id,
                            processing_fingerprint=(
                                fingerprint
                            ),
                        )
                    )

                    if (
                        existing_version
                        and existing_version.status
                        == ProcessingStatus.ACTIVE.value
                    ):
                        run = (
                            await IngestionRepository
                            .get_by_id(
                                session,
                                run_id,
                            )
                        )

                        await (
                            IngestionRepository
                            .increment_skipped(
                                session,
                                run,
                            )
                        )

                        await (
                            IngestionRepository
                            .complete_run(
                                session,
                                run,
                            )
                        )

                        log.info(
                            "filing_ingestion_skipped",
                            accession_number=(
                                accession_number
                            ),
                            reason=(
                                "processing_version_"
                                "already_active"
                            ),
                        )

                        return IngestionResult(
                            ingestion_run_id=run_id,
                            status=(
                                IngestionStatus.COMPLETED
                            ),
                            filings_discovered=1,
                            filings_processed=0,
                            filings_skipped=1,
                            filings_failed=0,
                        )

            # --------------------------------------
            # 5. Fetch filing content
            # --------------------------------------

            current_stage = IngestionStage.FETCH

            await self._set_stage(
                run_id,
                current_stage,
            )

            filing_content = (
                await self.edgar.fetch_content(
                    filing,
                    filing_id,
                )
            )

            # --------------------------------------
            # 6. Persist fetch metadata
            # --------------------------------------

            async with self.db.session() as session:

                filing_model = (
                    await FilingRepository.get_by_id(
                        session,
                        filing_id,
                    )
                )

                await FilingRepository.mark_fetched(
                    session,
                    filing_model,
                    content_hash=(
                        filing_content.content_hash
                    ),
                    source_size_bytes=(
                        filing_content.source_size_bytes
                    ),
                )

            # --------------------------------------
            # 7. Processing fingerprint
            # --------------------------------------

            fingerprint = (
                self._processing_fingerprint(
                    filing_content.content_hash
                )
            )

            # --------------------------------------
            # 8. Create / reset ProcessingVersion
            # --------------------------------------

            async with self.db.session() as session:

                existing_version = (
                    await ProcessingRepository
                    .get_by_fingerprint(
                        session,
                        filing_id=filing_id,
                        processing_fingerprint=(
                            fingerprint
                        ),
                    )
                )

                if (
                    existing_version
                    and existing_version.status
                    == ProcessingStatus.ACTIVE.value
                ):
                    run = await IngestionRepository.get_by_id(
                        session,
                        run_id,
                    )

                    await IngestionRepository.increment_skipped(
                        session,
                        run,
                    )

                    await IngestionRepository.complete_run(
                        session,
                        run,
                    )

                    return IngestionResult(
                        ingestion_run_id=run_id,
                        status=IngestionStatus.COMPLETED,
                        filings_discovered=1,
                        filings_processed=0,
                        filings_skipped=1,
                        filings_failed=0,
                    )

                if existing_version:

                    processing_version = (
                        await ProcessingRepository
                        .reset_version(
                            session,
                            existing_version,
                            run_id,
                        )
                    )

                else:

                    processing_schema = (
                        self._processing_schema(
                            filing_id=filing_id,
                            run_id=run_id,
                            fingerprint=fingerprint,
                        )
                    )

                    processing_version = (
                        await ProcessingRepository
                        .create_version(
                            session,
                            processing_schema,
                        )
                    )

                processing_version_id = (
                    processing_version.id
                )

            # --------------------------------------
            # 9. Extract SEC sections
            # --------------------------------------

            current_stage = IngestionStage.EXTRACT

            await self._set_stage(
                run_id,
                current_stage,
            )

            sections = (
                await self.edgar.extract_sections(
                    filing
                )
            )

            if not sections:
                raise ValueError(
                    "No SEC sections were extracted."
                )

            # --------------------------------------
            # 10. Normalize / chunk
            # --------------------------------------

            current_stage = IngestionStage.CHUNK

            await self._set_stage(
                run_id,
                current_stage,
            )

            source_url = str(
                filing.filing_url
                or filing.text_url
            )

            chunks = self.chunker.chunk(
                sections=sections,
                filing_id=filing_id,
                processing_version_id=(
                    processing_version_id
                ),
                source_url=source_url,
            )

            if not chunks:
                raise ValueError(
                    "Chunking produced zero chunks."
                )

            log.info(
                "filing_chunked",
                accession_number=accession_number,
                chunk_count=len(chunks),
            )

            # --------------------------------------
            # 11. Embeddings
            # --------------------------------------

            current_stage = IngestionStage.EMBED

            await self._set_stage(
                run_id,
                current_stage,
            )

            embedded_chunks = (
                await self.embedder.embed_chunks(
                    chunks
                )
            )

            if len(embedded_chunks) != len(chunks):
                raise ValueError(
                    "Embedding count does not match "
                    "chunk count."
                )

            # --------------------------------------
            # 12. Persist vectors
            # --------------------------------------

            current_stage = IngestionStage.PERSIST

            await self._set_stage(
                run_id,
                current_stage,
            )

            async with self.db.session() as session:

                await ProcessingRepository.add_chunks(
                    session,
                    embedded_chunks,
                )

            # --------------------------------------
            # 13. Validate + activate
            # --------------------------------------

            current_stage = IngestionStage.VALIDATE

            await self._set_stage(
                run_id,
                current_stage,
            )

            current_stage = IngestionStage.ACTIVATE

            await self._set_stage(
                run_id,
                current_stage,
            )

            async with self.db.session() as session:

                version_model = (
                    await ProcessingRepository.get_by_id(
                        session,
                        processing_version_id,
                    )
                )

                await ProcessingRepository.activate(
                    session,
                    version_model,
                )

                run = await IngestionRepository.get_by_id(
                    session,
                    run_id,
                )

                await IngestionRepository.increment_processed(
                    session,
                    run,
                )

                await IngestionRepository.complete_run(
                    session,
                    run,
                )

            log.info(
                "filing_ingestion_completed",
                accession_number=accession_number,
                filing_id=str(filing_id),
                processing_version_id=str(
                    processing_version_id
                ),
                chunks=len(embedded_chunks),
            )

            return IngestionResult(
                ingestion_run_id=run_id,
                status=IngestionStatus.COMPLETED,
                filings_discovered=1,
                filings_processed=1,
                filings_skipped=0,
                filings_failed=0,
                processed_filing_ids=[
                    filing_id
                ],
            )

        except Exception as exc:

            if run_id is not None:

                await self._record_failure(
                    run_id=run_id,
                    filing_id=filing_id,
                    processing_version_id=(
                        processing_version_id
                    ),
                    accession_number=(
                        accession_number
                    ),
                    stage=current_stage,
                    exc=exc,
                )

            raise

    async def _set_stage(
        self,
        run_id: UUID,
        stage: IngestionStage,
    ) -> None:

        async with self.db.session() as session:

            run = await IngestionRepository.get_by_id(
                session,
                run_id,
            )

            await IngestionRepository.set_stage(
                session,
                run,
                stage.value,
            )

    def _processing_fingerprint(
        self,
        content_hash: str,
    ) -> str:

        embedding = self.settings.embedding

        values = [
            content_hash,

            "edgartools",
            self.parser_version,

            self.NORMALIZATION_VERSION,

            "sec-section-aware",
            "v1",
            str(self.chunker.max_tokens),
            str(self.chunker.overlap_tokens),

            embedding.provider.value,
            embedding.model_name,
            str(embedding.dimension),

            DistanceMetric.COSINE.value,
        ]

        value = "|".join(values)

        return hashlib.sha256(
            value.encode("utf-8")
        ).hexdigest()

    def _processing_schema(
        self,
        *,
        filing_id: UUID,
        run_id: UUID,
        fingerprint: str,
    ) -> ProcessingVersionCreate:

        embedding = self.settings.embedding

        return ProcessingVersionCreate(
            filing_id=filing_id,

            ingestion_run_id=run_id,

            pipeline_version=(
                self.PIPELINE_VERSION
            ),

            parser_name="edgartools",

            parser_version=(
                self.parser_version
            ),

            normalization_version=(
                self.NORMALIZATION_VERSION
            ),

            chunking_strategy=(
                "sec-section-aware"
            ),

            chunking_version=(
                "v1-800-100"
            ),

            embedding_provider=(
                embedding.provider.value
            ),

            embedding_model=(
                embedding.model_name
            ),

            embedding_revision=None,

            embedding_dimension=(
                embedding.dimension
            ),

            embedding_normalized=None,

            embedding_instruction=None,

            distance_metric=(
                DistanceMetric.COSINE
            ),

            processing_fingerprint=(
                fingerprint
            ),

            index_version=(
                "pgvector-exact-v1"
            ),
        )

    async def _record_failure(
        self,
        *,
        run_id: UUID,
        filing_id: UUID | None,
        processing_version_id: UUID | None,
        accession_number: str | None,
        stage: IngestionStage,
        exc: Exception,
    ) -> None:

        try:
            async with self.db.session() as session:

                run = (
                    await IngestionRepository.get_by_id(
                        session,
                        run_id,
                    )
                )

                if run is not None:

                    await (
                        IngestionRepository
                        .increment_failed(
                            session,
                            run,
                        )
                    )

                    await IngestionRepository.add_error(
                        session,
                        IngestionError(
                            ingestion_run_id=run_id,

                            filing_id=filing_id,

                            accession_number=(
                                accession_number
                            ),

                            stage=stage,

                            error_type=(
                                type(exc).__name__
                            ),

                            message=str(exc),

                            retriable=False,

                            occurred_at=(
                                datetime.now(
                                    timezone.utc
                                )
                            ),
                        ),
                    )

                    await IngestionRepository.fail_run(
                        session,
                        run,
                    )

                if processing_version_id:

                    version_model = (
                        await ProcessingRepository
                        .get_by_id(
                            session,
                            processing_version_id,
                        )
                    )

                    if version_model is not None:

                        await (
                            ProcessingRepository
                            .mark_failed(
                                session,
                                version_model,
                            )
                        )

        except Exception as logging_exc:

            log.error(
                "failed_to_record_ingestion_error",
                original_error=str(exc),
                logging_error=str(logging_exc),
            )
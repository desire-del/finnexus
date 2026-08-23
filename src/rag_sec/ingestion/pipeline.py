import hashlib

from datetime import datetime, timezone
from importlib.metadata import version as package_version
from uuid import UUID

from rag_sec.config import get_settings

from rag_sec.database.manager import (
    get_database_manager,
)

from rag_sec.database.repositories import (
    CompanyRepository,
    FilingRepository,
    ProcessingRepository,
    IngestionRepository,
)

from rag_sec.ingestion.edgar_client import (
    EdgarClient,
)

from rag_sec.ingestion.chunker import (
    SectionChunker,
)

from rag_sec.providers import (
    get_embedding_model,
)

from rag_sec.logging import (
    get_logger,
)

from rag_sec.observability import (
    Phase,
    set_span_attributes,
    set_span_input,
    track,
)

from rag_sec.schemas.chunk import (
    EmbeddedChunk,
)

from rag_sec.schemas.filing import (
    FilingSection,
)

from rag_sec.schemas.processing import (
    ProcessingVersionCreate,
)

from rag_sec.schemas.ingestion import (
    IngestionError,
    IngestionResult,
    IngestionRunCreate,
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
    End-to-end ingestion pipeline for one SEC filing.

    Flow:
        SEC discovery
        -> filing registry
        -> fetch
        -> processing version
        -> section extraction
        -> normalization
        -> chunking
        -> embeddings
        -> pgvector persistence
        -> validation
        -> activation
    """

    PIPELINE_VERSION = "ingestion-v1"

    NORMALIZATION_VERSION = "normalize-v1"

    CHUNKING_STRATEGY = (
        "sec-sections+recursive-character"
    )

    CHUNKING_VERSION = "v1"

    INDEX_VERSION = "pgvector-exact-v1"

    def __init__(
        self,
        *,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ):
        self.settings = get_settings()

        self.db = get_database_manager()

        self.edgar = EdgarClient()

        self.chunker = SectionChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        self.embeddings = (
            get_embedding_model()
        )

        # Versions that may influence processing.
        self.edgartools_version = (
            package_version("edgartools")
        )

        self.text_splitters_version = (
            package_version(
                "langchain-text-splitters"
            )
        )

    @track(
        name="ingestion.latest_filing",
        phase=Phase.INGESTION,
        tags=["component:ingestion"],
    )
    async def ingest_latest(
        self,
        identifier: str | int,
        form_type: str = "10-K",
    ) -> IngestionResult:
        """
        Ingest the latest filing of a given SEC form
        for a company.

        Example:
            await pipeline.ingest_latest(
                "AAPL",
                form_type="10-K",
            )
        """

        run_id: UUID | None = None
        filing_id: UUID | None = None

        processing_version_id: (
            UUID | None
        ) = None

        accession_number: str | None = None

        current_stage = (
            IngestionStage.DISCOVER
        )

        set_span_attributes(
            {
                "rag.ingestion.identifier": str(identifier),
                "rag.ingestion.form_type": form_type,
                "rag.embedding.provider": (
                    self.settings.embedding.provider.value
                ),
                "rag.embedding.model": (
                    self.settings.embedding.model_name
                ),
                "rag.embedding.dimension": (
                    self.settings.embedding.dimension
                ),
            }
        )
        set_span_input(
            {
                "identifier": str(identifier),
                "form_type": form_type,
                "embedding_provider": (
                    self.settings.embedding.provider.value
                ),
                "embedding_model": (
                    self.settings.embedding.model_name
                ),
                "embedding_dimension": (
                    self.settings.embedding.dimension
                ),
            }
        )

        try:
            # ======================================
            # 1. INGESTION RUN
            # ======================================

            run_id = await self._create_run()

            # ======================================
            # 2. DISCOVER
            # ======================================

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

            log.info(
                "filing_discovered",
                identifier=str(identifier),
                accession_number=(
                    accession_number
                ),
                form_type=form_type,
            )

            # ======================================
            # 3. REGISTER COMPANY + FILING
            # ======================================

            company_schema = (
                self.edgar.to_company_schema(
                    company
                )
            )

            async with self.db.session() as session:

                run = (
                    await IngestionRepository
                    .get_by_id(
                        session,
                        run_id,
                    )
                )

                await (
                    IngestionRepository
                    .increment_discovered(
                        session,
                        run,
                    )
                )

                company_model = (
                    await CompanyRepository
                    .get_or_create(
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
                        self.edgar
                        .to_filing_schema(
                            filing,
                            company_model.id,
                        )
                    )

                    filing_model = (
                        await FilingRepository
                        .create(
                            session,
                            filing_schema,
                        )
                    )

                filing_id = filing_model.id

                known_content_hash = (
                    filing_model.content_hash
                )

            # ======================================
            # 4. FAST IDEMPOTENCY CHECK
            # ======================================

            if known_content_hash:

                fingerprint = (
                    self._processing_fingerprint(
                        known_content_hash
                    )
                )

                existing = (
                    await self
                    ._get_processing_version(
                        filing_id,
                        fingerprint,
                    )
                )

                if (
                    existing is not None
                    and existing.status
                    == ProcessingStatus.ACTIVE.value
                ):
                    return await self._skip(
                        run_id=run_id,
                        accession_number=(
                            accession_number
                        ),
                    )

            # ======================================
            # 5. FETCH
            # ======================================

            current_stage = (
                IngestionStage.FETCH
            )

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

            # Persist source hash/size.
            async with self.db.session() as session:

                filing_model = (
                    await FilingRepository
                    .get_by_id(
                        session,
                        filing_id,
                    )
                )

                await (
                    FilingRepository
                    .mark_fetched(
                        session,
                        filing_model,
                        content_hash=(
                            filing_content
                            .content_hash
                        ),
                        source_size_bytes=(
                            filing_content
                            .source_size_bytes
                        ),
                    )
                )

            # ======================================
            # 6. PROCESSING FINGERPRINT
            # ======================================

            fingerprint = (
                self._processing_fingerprint(
                    filing_content.content_hash
                )
            )

            # ======================================
            # 7. PROCESSING VERSION
            # ======================================

            processing_version_id = (
                await self
                ._prepare_processing_version(
                    filing_id=filing_id,
                    run_id=run_id,
                    fingerprint=fingerprint,
                )
            )

            # None means it already became ACTIVE.
            if processing_version_id is None:

                return await self._skip(
                    run_id=run_id,
                    accession_number=(
                        accession_number
                    ),
                )

            # ======================================
            # 8. EXTRACT STRUCTURED SEC SECTIONS
            # ======================================

            current_stage = (
                IngestionStage.EXTRACT
            )

            await self._set_stage(
                run_id,
                current_stage,
            )

            sections = (
                await self.edgar
                .extract_sections(
                    filing
                )
            )

            if not sections:
                raise ValueError(
                    "No sections extracted "
                    "from SEC filing."
                )

            log.info(
                "filing_sections_extracted",
                accession_number=(
                    accession_number
                ),
                section_count=len(sections),
            )

            # ======================================
            # 9. NORMALIZE
            # ======================================

            current_stage = (
                IngestionStage.NORMALIZE
            )

            await self._set_stage(
                run_id,
                current_stage,
            )

            sections = self._normalize_sections(
                sections
            )

            # ======================================
            # 10. CHUNK
            # ======================================

            current_stage = (
                IngestionStage.CHUNK
            )

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

            chunks = [
                chunk.model_copy(
                    update={
                        "metadata": {
                            **chunk.metadata,

                            "company_name":
                                company.name,

                            "ticker":
                                company.get_ticker(),

                            "accession_number":
                                accession_number,

                            "form_type":
                                filing.form,

                            "filing_date":
                                str(
                                    filing.filing_date
                                ),
                        }
                    }
                )
                for chunk in chunks
            ]



            if not chunks:
                raise ValueError(
                    "Chunking produced "
                    "zero chunks."
                )

            log.info(
                "filing_chunked",
                accession_number=(
                    accession_number
                ),
                chunk_count=len(chunks),
            )

            # ======================================
            # 11. EMBEDDINGS
            # ======================================

            current_stage = (
                IngestionStage.EMBED
            )

            await self._set_stage(
                run_id,
                current_stage,
            )

            texts = [
                chunk.text
                for chunk in chunks
            ]

            vectors = (
                await self.embeddings
                .aembed_documents(
                    texts
                )
            )

            embedded_chunks = (
                self._attach_embeddings(
                    chunks,
                    vectors,
                )
            )

            log.info(
                "filing_embedded",
                accession_number=(
                    accession_number
                ),
                chunk_count=(
                    len(embedded_chunks)
                ),
                embedding_model=(
                    self.settings
                    .embedding
                    .model_name
                ),
                embedding_dimension=(
                    self.settings
                    .embedding
                    .dimension
                ),
            )

            # ======================================
            # 12. PERSIST CHUNKS + VECTORS
            # ======================================

            current_stage = (
                IngestionStage.PERSIST
            )

            await self._set_stage(
                run_id,
                current_stage,
            )

            async with self.db.session() as session:

                await (
                    ProcessingRepository
                    .add_chunks(
                        session,
                        embedded_chunks,
                    )
                )

            # ======================================
            # 13. VALIDATE
            # ======================================

            current_stage = (
                IngestionStage.VALIDATE
            )

            await self._set_stage(
                run_id,
                current_stage,
            )

            await self._validate_processing(
                processing_version_id
            )

            # ======================================
            # 14. ACTIVATE
            # ======================================

            current_stage = (
                IngestionStage.ACTIVATE
            )

            await self._set_stage(
                run_id,
                current_stage,
            )

            async with self.db.session() as session:

                processing_version = (
                    await ProcessingRepository
                    .get_by_id(
                        session,
                        processing_version_id,
                    )
                )

                if processing_version is None:
                    raise RuntimeError(
                        "Processing version "
                        "not found before activation."
                    )

                await (
                    ProcessingRepository
                    .activate(
                        session,
                        processing_version,
                    )
                )

                run = (
                    await IngestionRepository
                    .get_by_id(
                        session,
                        run_id,
                    )
                )

                await (
                    IngestionRepository
                    .increment_processed(
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
                "filing_ingestion_completed",
                accession_number=(
                    accession_number
                ),
                filing_id=str(filing_id),
                processing_version_id=str(
                    processing_version_id
                ),
                chunk_count=(
                    len(embedded_chunks)
                ),
            )

            return IngestionResult(
                ingestion_run_id=run_id,

                status=(
                    IngestionStatus.COMPLETED
                ),

                filings_discovered=1,
                filings_processed=1,
                filings_skipped=0,
                filings_failed=0,

                processed_filing_ids=[
                    filing_id
                ],
            )

        except Exception as exc:

            log.exception(
                "filing_ingestion_failed",
                accession_number=(
                    accession_number
                ),
                stage=current_stage.value,
                error=str(exc),
            )

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

    # ==================================================
    # RUN
    # ==================================================

    async def _create_run(
        self,
    ) -> UUID:

        async with self.db.session() as session:

            run = (
                await IngestionRepository
                .create_run(
                    session,
                    IngestionRunCreate(
                        pipeline_version=(
                            self.PIPELINE_VERSION
                        )
                    ),
                )
            )

            return run.id

    async def _set_stage(
        self,
        run_id: UUID,
        stage: IngestionStage,
    ) -> None:

        set_span_attributes(
            {
                "rag.ingestion.stage": stage.value,
            }
        )

        async with self.db.session() as session:

            run = (
                await IngestionRepository
                .get_by_id(
                    session,
                    run_id,
                )
            )

            if run is None:
                raise RuntimeError(
                    "Ingestion run not found."
                )

            await (
                IngestionRepository
                .set_stage(
                    session,
                    run,
                    stage.value,
                )
            )

    # ==================================================
    # IDEMPOTENCY / PROCESSING VERSION
    # ==================================================

    def _processing_fingerprint(
        self,
        content_hash: str,
    ) -> str:
        """
        Unique representation of:

            source content
            + parser
            + normalization
            + chunking
            + embeddings
            + metric
        """

        embedding = (
            self.settings.embedding
        )

        values = [
            content_hash,

            "edgartools",
            self.edgartools_version,

            self.NORMALIZATION_VERSION,

            self.CHUNKING_STRATEGY,
            self.CHUNKING_VERSION,

            self.text_splitters_version,

            str(
                self.chunker.chunk_size
            ),

            str(
                self.chunker.chunk_overlap
            ),

            embedding.provider.value,

            embedding.model_name,

            str(
                embedding.dimension
            ),

            DistanceMetric.COSINE.value,
        ]

        raw = "|".join(values)

        return hashlib.sha256(
            raw.encode("utf-8")
        ).hexdigest()

    async def _get_processing_version(
        self,
        filing_id: UUID,
        fingerprint: str,
    ):
        async with self.db.session() as session:

            return (
                await ProcessingRepository
                .get_by_fingerprint(
                    session,

                    filing_id=filing_id,

                    processing_fingerprint=(
                        fingerprint
                    ),
                )
            )

    async def _prepare_processing_version(
        self,
        *,
        filing_id: UUID,
        run_id: UUID,
        fingerprint: str,
    ) -> UUID | None:
        """
        Return:
            UUID -> BUILDING version ready to use
            None -> identical ACTIVE version exists
        """

        async with self.db.session() as session:

            existing = (
                await ProcessingRepository
                .get_by_fingerprint(
                    session,

                    filing_id=filing_id,

                    processing_fingerprint=(
                        fingerprint
                    ),
                )
            )

            # Already successfully processed.
            if (
                existing is not None
                and existing.status
                == ProcessingStatus.ACTIVE.value
            ):
                return None

            # Previous attempt existed but failed
            # or remained incomplete.
            if existing is not None:

                version = (
                    await ProcessingRepository
                    .reset_version(
                        session,
                        processing_version_id=(
                            existing.id
                        ),
                        ingestion_run_id=run_id,
                    )
                )

                return version.id

            # New processing configuration.
            schema = (
                self._processing_schema(
                    filing_id=filing_id,
                    run_id=run_id,
                    fingerprint=fingerprint,
                )
            )

            version = (
                await ProcessingRepository
                .create_version(
                    session,
                    schema,
                )
            )

            return version.id

    def _processing_schema(
        self,
        *,
        filing_id: UUID,
        run_id: UUID,
        fingerprint: str,
    ) -> ProcessingVersionCreate:

        embedding = (
            self.settings.embedding
        )

        return ProcessingVersionCreate(
            filing_id=filing_id,

            ingestion_run_id=run_id,

            pipeline_version=(
                self.PIPELINE_VERSION
            ),

            parser_name="edgartools",

            parser_version=(
                self.edgartools_version
            ),

            normalization_version=(
                self.NORMALIZATION_VERSION
            ),

            chunking_strategy=(
                self.CHUNKING_STRATEGY
            ),

            chunking_version=(
                self.CHUNKING_VERSION
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
                self.INDEX_VERSION
            ),
        )

    # ==================================================
    # NORMALIZATION
    # ==================================================

    @staticmethod
    def _normalize_sections(
        sections: list[FilingSection],
    ) -> list[FilingSection]:
        """
        Conservative normalization only.

        Financial values and Markdown structure
        must not be modified.
        """

        normalized = []

        for section in sections:

            content = (
                section.content
                .replace("\r\n", "\n")
                .replace("\r", "\n")
                .strip()
            )

            if not content:
                continue

            normalized.append(
                section.model_copy(
                    update={
                        "content": content
                    }
                )
            )

        return normalized

    # ==================================================
    # EMBEDDINGS
    # ==================================================

    def _attach_embeddings(
        self,
        chunks,
        vectors: list[list[float]],
    ) -> list[EmbeddedChunk]:

        if len(chunks) != len(vectors):
            raise ValueError(
                "Embedding count does not match "
                "chunk count."
            )

        expected_dimension = (
            self.settings
            .embedding
            .dimension
        )

        embedded_chunks = []

        for chunk, vector in zip(
            chunks,
            vectors,
        ):
            if (
                len(vector)
                != expected_dimension
            ):
                raise ValueError(
                    "Unexpected embedding "
                    f"dimension: {len(vector)}. "
                    f"Expected "
                    f"{expected_dimension}."
                )

            embedded_chunks.append(
                EmbeddedChunk(
                    **chunk.model_dump(),
                    embedding=vector,
                )
            )

        return embedded_chunks

    # ==================================================
    # VALIDATION
    # ==================================================

    async def _validate_processing(
        self,
        processing_version_id: UUID,
    ) -> None:
        """
        Minimal validation before activation.

        More sophisticated quality checks can
        be added later.
        """

        async with self.db.session() as session:

            chunk_count = (
                await ProcessingRepository
                .count_chunks(
                    session,
                    processing_version_id,
                )
            )

            if chunk_count == 0:
                raise ValueError(
                    "Processing version contains "
                    "zero chunks."
                )

        log.info(
            "processing_version_validated",
            processing_version_id=str(
                processing_version_id
            ),
            chunk_count=chunk_count,
        )

    # ==================================================
    # SKIP
    # ==================================================

    async def _skip(
        self,
        *,
        run_id: UUID,
        accession_number: str,
    ) -> IngestionResult:

        async with self.db.session() as session:

            run = (
                await IngestionRepository
                .get_by_id(
                    session,
                    run_id,
                )
            )

            if run is None:
                raise RuntimeError(
                    "Ingestion run not found."
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
                "processing_version_already_active"
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

    # ==================================================
    # FAILURE
    # ==================================================

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
        """
        Persist failure information without hiding
        the original exception.
        """

        try:
            async with self.db.session() as session:

                run = (
                    await IngestionRepository
                    .get_by_id(
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

                    await (
                        IngestionRepository
                        .add_error(
                            session,
                            IngestionError(
                                ingestion_run_id=(
                                    run_id
                                ),

                                filing_id=(
                                    filing_id
                                ),

                                accession_number=(
                                    accession_number
                                ),

                                stage=stage,

                                error_type=(
                                    type(exc)
                                    .__name__
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
                    )

                    await (
                        IngestionRepository
                        .fail_run(
                            session,
                            run,
                        )
                    )

                if (
                    processing_version_id
                    is not None
                ):
                    version = (
                        await ProcessingRepository
                        .get_by_id(
                            session,
                            processing_version_id,
                        )
                    )

                    if version is not None:

                        await (
                            ProcessingRepository
                            .mark_failed(
                                session,
                                version,
                            )
                        )

        except Exception as failure_exc:

            log.error(
                "failed_to_record_ingestion_failure",
                original_error=str(exc),
                persistence_error=str(
                    failure_exc
                ),
            )

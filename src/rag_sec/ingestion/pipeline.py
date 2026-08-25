from uuid import UUID

from rag_sec.config import get_settings
from rag_sec.database.company_repository import CompanyRepository
from rag_sec.database.filing_repository import FilingRepository
from rag_sec.database.ingestion_repository import IngestionRepository
from rag_sec.database.manager import (
    get_database_manager,
)
from rag_sec.database.processing_repository import ProcessingRepository
from rag_sec.ingestion.chunker import (
    SectionChunker,
)
from rag_sec.ingestion.edgar_client import (
    EdgarClient,
)
from rag_sec.ingestion.processing import ProcessingProfile
from rag_sec.ingestion.run_tracker import IngestionRunTracker
from rag_sec.logging import (
    get_logger,
)
from rag_sec.observability import (
    Phase,
    set_span_attributes,
    set_span_input,
    track,
)
from rag_sec.providers import (
    get_embedding_model,
)
from rag_sec.schemas.enums import (
    IngestionStage,
    IngestionStatus,
    ProcessingStatus,
)
from rag_sec.schemas.ingestion import IngestionResult
from rag_sec.schemas.processing import ProcessingVersionCreate

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

    def __init__(
        self,
        *,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ) -> None:
        self.settings = get_settings()

        self.db = get_database_manager()

        self.edgar = EdgarClient()

        self.chunker = SectionChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        self.embeddings = get_embedding_model()

        self.processing = ProcessingProfile(self.settings.embedding, self.chunker)
        self.runs = IngestionRunTracker(
            self.db,
            pipeline_version=self.processing.pipeline_version,
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

        run_id = await self._create_run()

        set_span_attributes(
            {
                "rag.ingestion.mode": "latest",
                "rag.ingestion.identifier": str(identifier),
                "rag.ingestion.form_type": form_type,
                "rag.embedding.provider": (self.settings.embedding.provider.value),
                "rag.embedding.model": (self.settings.embedding.model_name),
                "rag.embedding.dimension": (self.settings.embedding.dimension),
            }
        )
        set_span_input(
            {
                "mode": "latest",
                "identifier": str(identifier),
                "form_type": form_type,
                "embedding_provider": (self.settings.embedding.provider.value),
                "embedding_model": (self.settings.embedding.model_name),
                "embedding_dimension": (self.settings.embedding.dimension),
            }
        )

        await self._set_stage(
            run_id,
            IngestionStage.DISCOVER,
        )

        try:
            company, filing = await self.edgar.get_latest_filing(
                identifier,
                form_type=form_type,
            )
        except Exception as exc:
            log.exception(
                "filing_discovery_failed",
                mode="latest",
                identifier=str(identifier),
                form_type=form_type,
                error=str(exc),
            )

            await self._record_failure(
                run_id=run_id,
                filing_id=None,
                processing_version_id=None,
                accession_number=None,
                stage=IngestionStage.DISCOVER,
                exc=exc,
            )
            raise

        log.info(
            "filing_discovered",
            mode="latest",
            identifier=str(identifier),
            accession_number=filing.accession_number,
            form_type=filing.form,
        )

        return await self._ingest_discovered_filing(
            run_id=run_id,
            company=company,
            filing=filing,
        )

    @track(
        name="ingestion.accession_filing",
        phase=Phase.INGESTION,
        tags=["component:ingestion"],
    )
    async def ingest_accession(
        self,
        accession_number: str,
    ) -> IngestionResult:
        """Ingest one exact SEC filing by accession number."""
        normalized_accession = accession_number.strip()

        if not normalized_accession:
            raise ValueError("Accession number cannot be empty.")

        run_id = await self._create_run()

        set_span_attributes(
            {
                "rag.ingestion.mode": "accession",
                "rag.ingestion.accession_number": normalized_accession,
                "rag.embedding.provider": (self.settings.embedding.provider.value),
                "rag.embedding.model": (self.settings.embedding.model_name),
                "rag.embedding.dimension": (self.settings.embedding.dimension),
            }
        )
        set_span_input(
            {
                "mode": "accession",
                "accession_number": normalized_accession,
                "embedding_provider": (self.settings.embedding.provider.value),
                "embedding_model": (self.settings.embedding.model_name),
                "embedding_dimension": (self.settings.embedding.dimension),
            }
        )

        await self._set_stage(
            run_id,
            IngestionStage.DISCOVER,
        )

        try:
            company, filing = await self.edgar.get_filing_by_accession(
                normalized_accession
            )
        except Exception as exc:
            log.exception(
                "filing_discovery_failed",
                mode="accession",
                accession_number=normalized_accession,
                error=str(exc),
            )

            await self._record_failure(
                run_id=run_id,
                filing_id=None,
                processing_version_id=None,
                accession_number=normalized_accession,
                stage=IngestionStage.DISCOVER,
                exc=exc,
            )
            raise

        log.info(
            "filing_discovered",
            mode="accession",
            accession_number=filing.accession_number,
            form_type=filing.form,
        )

        return await self._ingest_discovered_filing(
            run_id=run_id,
            company=company,
            filing=filing,
        )

    async def _ingest_discovered_filing(
        self,
        *,
        run_id: UUID,
        company,
        filing,
    ) -> IngestionResult:
        """Process a filing discovered by any supported ingestion mode."""
        filing_id: UUID | None = None
        processing_version_id: UUID | None = None
        accession_number = filing.accession_number
        current_stage = IngestionStage.DISCOVER

        set_span_attributes(
            {
                "rag.ingestion.accession_number": accession_number,
                "rag.ingestion.form_type": filing.form,
                "rag.ingestion.cik": str(filing.cik),
            }
        )

        try:
            # ======================================
            # 3. REGISTER COMPANY + FILING
            # ======================================

            company_schema = self.edgar.to_company_schema(company)

            async with self.db.session() as session:
                run = await IngestionRepository.get_by_id(
                    session,
                    run_id,
                )
                if run is None:
                    raise RuntimeError("Ingestion run not found.")

                await IngestionRepository.increment_discovered(
                    session,
                    run,
                )

                company_model = await CompanyRepository.get_or_create(
                    session,
                    company_schema,
                )

                filing_model = await FilingRepository.get_by_accession_number(
                    session,
                    accession_number,
                )

                if filing_model is None:
                    filing_schema = self.edgar.to_filing_schema(
                        filing,
                        company_model.id,
                    )

                    filing_model = await FilingRepository.create(
                        session,
                        filing_schema,
                    )

                filing_id = filing_model.id

                known_content_hash = filing_model.content_hash

            # ======================================
            # 4. FAST IDEMPOTENCY CHECK
            # ======================================

            if known_content_hash:
                fingerprint = self.processing.fingerprint(known_content_hash)

                existing = await self._get_processing_version(
                    filing_id,
                    fingerprint,
                )

                if (
                    existing is not None
                    and existing.status == ProcessingStatus.ACTIVE.value
                ):
                    return await self._skip(
                        run_id=run_id,
                        accession_number=(accession_number),
                    )

            # ======================================
            # 5. FETCH
            # ======================================

            current_stage = IngestionStage.FETCH

            await self._set_stage(
                run_id,
                current_stage,
            )

            filing_content = await self.edgar.fetch_content(
                filing,
                filing_id,
            )

            # Persist source hash/size.
            async with self.db.session() as session:
                filing_model = await FilingRepository.get_by_id(
                    session,
                    filing_id,
                )
                if filing_model is None:
                    raise RuntimeError("Filing not found after content fetch.")

                await FilingRepository.mark_fetched(
                    session,
                    filing_model,
                    content_hash=(filing_content.content_hash),
                    source_size_bytes=(filing_content.source_size_bytes),
                )

            # ======================================
            # 6. PROCESSING FINGERPRINT
            # ======================================

            fingerprint = self.processing.fingerprint(filing_content.content_hash)

            # ======================================
            # 7. PROCESSING VERSION
            # ======================================

            processing_version_id = await self._prepare_processing_version(
                filing_id=filing_id,
                run_id=run_id,
                fingerprint=fingerprint,
            )

            # None means it already became ACTIVE.
            if processing_version_id is None:
                return await self._skip(
                    run_id=run_id,
                    accession_number=(accession_number),
                )

            # ======================================
            # 8. EXTRACT STRUCTURED SEC SECTIONS
            # ======================================

            current_stage = IngestionStage.EXTRACT

            await self._set_stage(
                run_id,
                current_stage,
            )

            sections = await self.edgar.extract_sections(filing)

            if not sections:
                raise ValueError("No sections extracted from SEC filing.")

            log.info(
                "filing_sections_extracted",
                accession_number=(accession_number),
                section_count=len(sections),
            )

            # ======================================
            # 9. NORMALIZE
            # ======================================

            current_stage = IngestionStage.NORMALIZE

            await self._set_stage(
                run_id,
                current_stage,
            )

            sections = self.processing.normalize_sections(sections)

            # ======================================
            # 10. CHUNK
            # ======================================

            current_stage = IngestionStage.CHUNK

            await self._set_stage(
                run_id,
                current_stage,
            )

            source_url = str(filing.filing_url or filing.text_url)

            chunks = self.chunker.chunk(
                sections=sections,
                filing_id=filing_id,
                processing_version_id=(processing_version_id),
                source_url=source_url,
            )

            chunks = [
                chunk.model_copy(
                    update={
                        "metadata": {
                            **chunk.metadata,
                            "company_name": company.name,
                            "ticker": company.get_ticker(),
                            "accession_number": accession_number,
                            "form_type": filing.form,
                            "filing_date": str(filing.filing_date),
                        }
                    }
                )
                for chunk in chunks
            ]

            if not chunks:
                raise ValueError("Chunking produced zero chunks.")

            log.info(
                "filing_chunked",
                accession_number=(accession_number),
                chunk_count=len(chunks),
            )

            # ======================================
            # 11. EMBEDDINGS
            # ======================================

            current_stage = IngestionStage.EMBED

            await self._set_stage(
                run_id,
                current_stage,
            )

            texts = [chunk.text for chunk in chunks]

            vectors = await self.embeddings.aembed_documents(texts)

            embedded_chunks = self.processing.attach_embeddings(
                chunks,
                vectors,
            )

            log.info(
                "filing_embedded",
                accession_number=(accession_number),
                chunk_count=(len(embedded_chunks)),
                embedding_model=(self.settings.embedding.model_name),
                embedding_dimension=(self.settings.embedding.dimension),
            )

            # ======================================
            # 12. PERSIST CHUNKS + VECTORS
            # ======================================

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

            # ======================================
            # 13. VALIDATE
            # ======================================

            current_stage = IngestionStage.VALIDATE

            await self._set_stage(
                run_id,
                current_stage,
            )

            await self._validate_processing(processing_version_id)

            # ======================================
            # 14. ACTIVATE
            # ======================================

            current_stage = IngestionStage.ACTIVATE

            await self._set_stage(
                run_id,
                current_stage,
            )

            async with self.db.session() as session:
                processing_version = await ProcessingRepository.get_by_id(
                    session,
                    processing_version_id,
                )

                if processing_version is None:
                    raise RuntimeError(
                        "Processing version not found before activation."
                    )

                await ProcessingRepository.activate(
                    session,
                    processing_version,
                )

                run = await IngestionRepository.get_by_id(
                    session,
                    run_id,
                )
                if run is None:
                    raise RuntimeError("Ingestion run not found before completion.")

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
                accession_number=(accession_number),
                filing_id=str(filing_id),
                processing_version_id=str(processing_version_id),
                chunk_count=(len(embedded_chunks)),
            )

            return IngestionResult(
                ingestion_run_id=run_id,
                status=(IngestionStatus.COMPLETED),
                filings_discovered=1,
                filings_processed=1,
                filings_skipped=0,
                filings_failed=0,
                processed_filing_ids=[filing_id],
            )

        except Exception as exc:
            log.exception(
                "filing_ingestion_failed",
                accession_number=(accession_number),
                stage=current_stage.value,
                error=str(exc),
            )

            if run_id is not None:
                await self._record_failure(
                    run_id=run_id,
                    filing_id=filing_id,
                    processing_version_id=(processing_version_id),
                    accession_number=(accession_number),
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
        return await self.runs.create()

    async def _set_stage(
        self,
        run_id: UUID,
        stage: IngestionStage,
    ) -> None:
        await self.runs.set_stage(run_id, stage)

    # ==================================================
    # IDEMPOTENCY / PROCESSING VERSION
    # ==================================================

    async def _get_processing_version(
        self,
        filing_id: UUID,
        fingerprint: str,
    ):
        async with self.db.session() as session:
            return await ProcessingRepository.get_by_fingerprint(
                session,
                filing_id=filing_id,
                processing_fingerprint=(fingerprint),
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
            existing = await ProcessingRepository.get_by_fingerprint(
                session,
                filing_id=filing_id,
                processing_fingerprint=(fingerprint),
            )

            # Already successfully processed.
            if (
                existing is not None
                and existing.status == ProcessingStatus.ACTIVE.value
            ):
                return None

            # Previous attempt existed but failed
            # or remained incomplete.
            if existing is not None:
                version = await ProcessingRepository.reset_version(
                    session,
                    processing_version_id=(existing.id),
                    ingestion_run_id=run_id,
                )

                return version.id

            # New processing configuration.
            schema = self._processing_schema(
                filing_id=filing_id,
                run_id=run_id,
                fingerprint=fingerprint,
            )

            version = await ProcessingRepository.create_version(
                session,
                schema,
            )

            return version.id

    def _processing_schema(
        self,
        *,
        filing_id: UUID,
        run_id: UUID,
        fingerprint: str,
    ) -> ProcessingVersionCreate:
        """Build the version schema through the active processing profile."""
        return self.processing.version_schema(
            filing_id=filing_id,
            ingestion_run_id=run_id,
            fingerprint=fingerprint,
        )

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
            chunk_count = await ProcessingRepository.count_chunks(
                session,
                processing_version_id,
            )

            if chunk_count == 0:
                raise ValueError("Processing version contains zero chunks.")

        log.info(
            "processing_version_validated",
            processing_version_id=str(processing_version_id),
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
        return await self.runs.complete_skipped(
            run_id,
            accession_number=accession_number,
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
        await self.runs.record_failure(
            run_id=run_id,
            filing_id=filing_id,
            processing_version_id=processing_version_id,
            accession_number=accession_number,
            stage=stage,
            error=exc,
        )

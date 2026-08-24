from langchain_core.documents import Document
from sqlalchemy import func, select

from rag_sec.database.manager import DatabaseManager
from rag_sec.models.chunk import Chunk
from rag_sec.models.company import Company
from rag_sec.models.filing import Filing
from rag_sec.models.processing_version import ProcessingVersion
from rag_sec.schemas.enums import ProcessingStatus


class BM25Store:
    """Search active chunks with ParadeDB's PostgreSQL-native BM25 index."""

    def __init__(
        self,
        database: DatabaseManager,
        *,
        embedding_provider: str,
        embedding_model: str,
        embedding_dimension: int,
    ) -> None:
        self.database = database
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.embedding_dimension = embedding_dimension

    async def search(
        self,
        query: str,
        *,
        ticker: str | None = None,
        form_type: str | None = None,
        accession_number: str | None = None,
        top_k: int,
    ) -> list[Document]:
        versions_statement = (
            select(ProcessingVersion.id, Filing, Company)
            .join(Filing, Filing.id == ProcessingVersion.filing_id)
            .join(Company, Company.id == Filing.company_id)
            .where(
                ProcessingVersion.status == ProcessingStatus.ACTIVE.value,
                ProcessingVersion.embedding_provider == self.embedding_provider,
                ProcessingVersion.embedding_model == self.embedding_model,
                ProcessingVersion.embedding_dimension == self.embedding_dimension,
            )
        )
        if ticker:
            versions_statement = versions_statement.where(
                Company.ticker == ticker.upper()
            )
        if form_type:
            versions_statement = versions_statement.where(
                Filing.form_type == form_type
            )
        if accession_number:
            versions_statement = versions_statement.where(
                Filing.accession_number == accession_number
            )

        async with self.database.session() as session:
            version_rows = (await session.execute(versions_statement)).all()
            versions = {
                processing_version_id: (filing, company)
                for processing_version_id, filing, company in version_rows
            }
            if not versions:
                return []

            score = func.pdb.score(Chunk.id).label("bm25_score")
            statement = (
                select(Chunk, score)
                .where(
                    Chunk.processing_version_id.in_(versions),
                    Chunk.embedding.is_not(None),
                    Chunk.text.op("|||")(query),
                )
                .order_by(score.desc(), Chunk.chunk_index.asc())
                .limit(top_k)
            )
            rows = (await session.execute(statement)).all()

        return [
            self._to_document(
                chunk,
                *versions[chunk.processing_version_id],
                float(score_value),
            )
            for chunk, score_value in rows
        ]

    def _to_document(
        self,
        chunk: Chunk,
        filing: Filing,
        company: Company,
        score: float,
    ) -> Document:
        return Document(
            page_content=chunk.text,
            metadata={
                **chunk.metadata_,
                "id": str(chunk.id),
                "chunk_id": chunk.chunk_id,
                "filing_id": str(chunk.filing_id),
                "processing_version_id": str(chunk.processing_version_id),
                "chunk_index": chunk.chunk_index,
                "section": chunk.section,
                "part": chunk.part,
                "item": chunk.item,
                "page": chunk.page,
                "source_url": chunk.source_url,
                "token_count": chunk.token_count,
                "accession_number": filing.accession_number,
                "form_type": filing.form_type,
                "filing_date": filing.filing_date,
                "cik": company.cik,
                "company_name": company.name,
                "ticker": company.ticker,
                "embedding_provider": self.embedding_provider,
                "embedding_model": self.embedding_model,
                "embedding_dimension": self.embedding_dimension,
                "bm25_score": score,
            },
        )

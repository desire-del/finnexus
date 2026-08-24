from langchain_core.documents import Document
from sqlalchemy import func, literal_column, select

from rag_sec.database.manager import DatabaseManager
from rag_sec.models.chunk import Chunk
from rag_sec.models.company import Company
from rag_sec.models.filing import Filing
from rag_sec.models.processing_version import ProcessingVersion
from rag_sec.schemas.enums import ProcessingStatus

ENGLISH_REGCONFIG = literal_column("'pg_catalog.english'::regconfig")


def disjunctive_websearch_query(query: str) -> str:
    """Let PostgreSQL parse terms while avoiding an all-terms-required query."""
    return " OR ".join(query.split())


class PostgresFTSStore:
    """Search active chunks with PostgreSQL native full-text search."""

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
        query_expression = func.websearch_to_tsquery(
            ENGLISH_REGCONFIG,
            disjunctive_websearch_query(query),
        )
        text_vector = func.to_tsvector(ENGLISH_REGCONFIG, Chunk.text)
        rank = func.ts_rank_cd(text_vector, query_expression).label("fts_rank")
        statement = (
            select(Chunk, Filing, Company, rank)
            .join(Filing, Filing.id == Chunk.filing_id)
            .join(Company, Company.id == Filing.company_id)
            .join(
                ProcessingVersion,
                ProcessingVersion.id == Chunk.processing_version_id,
            )
            .where(
                ProcessingVersion.status == ProcessingStatus.ACTIVE.value,
                ProcessingVersion.embedding_provider == self.embedding_provider,
                ProcessingVersion.embedding_model == self.embedding_model,
                ProcessingVersion.embedding_dimension == self.embedding_dimension,
                Chunk.embedding.is_not(None),
                text_vector.op("@@")(query_expression),
            )
            .order_by(rank.desc(), Chunk.chunk_index.asc())
            .limit(top_k)
        )
        if ticker:
            statement = statement.where(Company.ticker == ticker.upper())
        if form_type:
            statement = statement.where(Filing.form_type == form_type)
        if accession_number:
            statement = statement.where(Filing.accession_number == accession_number)

        async with self.database.session() as session:
            rows = (await session.execute(statement)).all()

        return [
            self._to_document(chunk, filing, company, float(rank_value))
            for chunk, filing, company, rank_value in rows
        ]

    def _to_document(
        self,
        chunk: Chunk,
        filing: Filing,
        company: Company,
        rank: float,
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
                "fts_rank": rank,
            },
        )

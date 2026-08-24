from langchain_core.documents import Document
from sqlalchemy import select

from rag_sec.database.manager import DatabaseManager
from rag_sec.models.chunk import Chunk
from rag_sec.models.company import Company
from rag_sec.models.filing import Filing
from rag_sec.models.processing_version import ProcessingVersion
from rag_sec.retrieval.bm25 import rank_bm25
from rag_sec.schemas.enums import ProcessingStatus


class BM25Store:
    """Load a filtered active-chunk corpus and rank it with Okapi BM25."""

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
        statement = (
            select(Chunk, Filing, Company)
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
            )
        )
        if ticker:
            statement = statement.where(Company.ticker == ticker.upper())
        if form_type:
            statement = statement.where(Filing.form_type == form_type)
        if accession_number:
            statement = statement.where(Filing.accession_number == accession_number)

        async with self.database.session() as session:
            rows = (await session.execute(statement)).all()

        documents = [
            self._to_document(chunk, filing, company) for chunk, filing, company in rows
        ]
        return [result.document for result in rank_bm25(query, documents, top_k=top_k)]

    def _to_document(
        self,
        chunk: Chunk,
        filing: Filing,
        company: Company,
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
            },
        )

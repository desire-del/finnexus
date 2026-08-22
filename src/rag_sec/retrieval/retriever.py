# src/rag_sec/retrieval.py

from langchain_core.documents import Document
from langchain_postgres import (
    PGEngine,
    PGVectorStore,
)
from langchain_postgres.v2.hybrid_search_config import (
    HybridSearchConfig,
    reciprocal_rank_fusion,
)

from rag_sec.database.manager import (
    get_database_manager,
)
from rag_sec.ingestion.embeddings import (
    get_embedding_model,
)
from rag_sec.logging import (
    get_logger,
)

log = get_logger(__name__)


class Retriever:

    def __init__(
        self,
        top_k: int = 5,
        dense_top_k: int = 20,
        lexical_top_k: int = 20,
    ):
        self.db = get_database_manager()

        self.embeddings = get_embedding_model()

        self.top_k = top_k
        self.dense_top_k = dense_top_k
        self.lexical_top_k = lexical_top_k

        self.vector_store: PGVectorStore | None = None

    async def initialize(self) -> None:

        if self.vector_store is not None:
            return

        pg_engine = PGEngine.from_engine(
            engine=self.db.engine
        )

        self.vector_store = await PGVectorStore.create(
            engine=pg_engine,

            embedding_service=self.embeddings,

            table_name="active_chunks",

            id_column="id",
            content_column="text",
            embedding_column="embedding",

            metadata_columns=[
                "chunk_id",
                "filing_id",
                "processing_version_id",
                "chunk_index",

                "section",
                "part",
                "item",
                "page",
                "source_url",
                "token_count",

                "accession_number",
                "form_type",
                "filing_date",

                "cik",
                "company_name",
                "ticker",
            ],

            metadata_json_column="metadata",
        )

        log.info(
            "retriever_initialized"
        )

    async def search(
        self,
        query: str,
        *,
        ticker: str | None = None,
        form_type: str | None = None,
        accession_number: str | None = None,
        top_k: int | None = None,
    ) -> list[Document]:

        query = query.strip()

        if not query:
            raise ValueError(
                "Query cannot be empty."
            )

        await self.initialize()

        filters = self._build_filters(
            ticker=ticker,
            form_type=form_type,
            accession_number=accession_number,
        )

        hybrid_config = HybridSearchConfig(
            tsv_lang="pg_catalog.english",

            fusion_function=(
                reciprocal_rank_fusion
            ),

            primary_top_k=(
                self.dense_top_k
            ),

            secondary_top_k=(
                self.lexical_top_k
            ),

            fusion_function_parameters={
                "rrf_k": 60
            },
        )

        documents = (
            await self.vector_store
            .asimilarity_search(
                query=query,
                k=top_k or self.top_k,
                filter=filters,
                hybrid_search_config=(
                    hybrid_config
                ),
            )
        )

        log.info(
            "hybrid_retrieval_completed",
            query=query,
            result_count=len(documents),
            ticker=ticker,
            form_type=form_type,
        )

        return documents

    @staticmethod
    def _build_filters(
        *,
        ticker: str | None,
        form_type: str | None,
        accession_number: str | None,
    ) -> dict | None:

        filters = []

        if ticker:
            filters.append(
                {
                    "ticker":
                        ticker.upper()
                }
            )

        if form_type:
            filters.append(
                {
                    "form_type":
                        form_type
                }
            )

        if accession_number:
            filters.append(
                {
                    "accession_number":
                        accession_number
                }
            )

        if not filters:
            return None

        if len(filters) == 1:
            return filters[0]

        return {
            "$and": filters
        }
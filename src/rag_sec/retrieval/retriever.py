# src/rag_sec/retrieval.py

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_postgres import (
    PGEngine,
    PGVectorStore,
)
from langchain_postgres.v2.hybrid_search_config import (
    HybridSearchConfig,
    reciprocal_rank_fusion,
)
from openinference.semconv.trace import (
    OpenInferenceSpanKindValues,
)

from rag_sec.database.manager import (
    get_database_manager,
)
from rag_sec.logging import (
    get_logger,
)
from rag_sec.observability import (
    Phase,
    set_span_attributes,
    set_span_input,
    set_span_output,
    track,
)

log = get_logger(__name__)


class Retriever:

    def __init__(
        self,
        embeddings: Embeddings,
        top_k: int = 5,
        dense_top_k: int = 20,
        lexical_top_k: int = 20,
    ):
        self.db = get_database_manager()

        self.embeddings = embeddings

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

    @track(
        name="retrieval.search",
        phase=Phase.RETRIEVAL,
        tags=["component:retriever"],
        span_kind=OpenInferenceSpanKindValues.RETRIEVER,
    )
    async def search(
        self,
        query: str,
        *,
        query_embedding: list[float],
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

        if not query_embedding:
            raise ValueError(
                "Query embedding cannot be empty."
            )

        result_limit = top_k or self.top_k

        set_span_attributes(
            {
                "rag.retrieval.query_length": len(query),
                "rag.retrieval.top_k": result_limit,
                "rag.retrieval.dense_top_k": self.dense_top_k,
                "rag.retrieval.lexical_top_k": self.lexical_top_k,
                "rag.retrieval.embedding_dimension": (
                    len(query_embedding)
                ),
                "rag.retrieval.ticker": ticker,
                "rag.retrieval.form_type": form_type,
                "rag.embedding.provider": (
                    self.embeddings.__class__.__name__
                ),
            }
        )
        set_span_input(
            {
                "query_length": len(query),
                "embedding_dimension": len(query_embedding),
                "ticker": ticker,
                "form_type": form_type,
                "accession_number": accession_number,
                "top_k": result_limit,
            }
        )

        if self.vector_store is None:
            raise RuntimeError(
                "Retriever is not ready. "
                "Run the application warmup first."
            )

        filters = self._build_filters(
            ticker=ticker,
            form_type=form_type,
            accession_number=accession_number,
        )

        hybrid_config = HybridSearchConfig(
            tsv_lang="pg_catalog.english",

            fts_query=query,

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
            .asimilarity_search_by_vector(
                embedding=query_embedding,
                k=result_limit,
                filter=filters,
                hybrid_search_config=(
                    hybrid_config
                ),
            )
        )

        set_span_attributes(
            {
                "rag.retrieval.result_count": len(documents),
            }
        )
        set_span_output(
            {
                "document_count": len(documents),
            }
        )

        log.info(
            "hybrid_retrieval_completed",
            query_length=len(query),
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

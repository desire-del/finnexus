# src/rag_sec/retrieval.py

from collections.abc import Sequence
from typing import Any, Literal

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

from rag_sec.config import get_settings
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
from rag_sec.retrieval.bm25 import reciprocal_rank_fuse_documents
from rag_sec.retrieval.bm25_store import BM25Store

log = get_logger(__name__)

RetrievalMode = Literal[
    "dense",
    "lexical",
    "hybrid",
    "bm25",
    "bm25_hybrid",
]


def lexical_only_ranking(
    primary_search_results: Sequence[Any],
    secondary_search_results: Sequence[Any],
    fetch_top_k: int = 4,
    **_: Any,
) -> Sequence[Any]:
    """Return PostgreSQL full-text results without dense contributions."""
    del primary_search_results
    return sorted(
        secondary_search_results,
        key=lambda result: result["distance"],
        reverse=True,
    )[:fetch_top_k]


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

        embedding_settings = get_settings().embedding
        self.embedding_provider = embedding_settings.provider.value
        self.embedding_model = embedding_settings.model_name
        self.embedding_dimension = embedding_settings.dimension
        self.bm25_store = BM25Store(
            self.db,
            embedding_provider=self.embedding_provider,
            embedding_model=self.embedding_model,
            embedding_dimension=self.embedding_dimension,
        )

        self.top_k = top_k
        self.dense_top_k = dense_top_k
        self.lexical_top_k = lexical_top_k

        self.vector_store: PGVectorStore | None = None

    async def initialize(self) -> None:

        if self.vector_store is not None:
            return

        pg_engine = PGEngine.from_engine(engine=self.db.engine)

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
                "embedding_provider",
                "embedding_model",
                "embedding_dimension",
            ],
            metadata_json_column="metadata",
        )

        log.info("retriever_initialized")

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
        query_embedding: list[float] | None = None,
        ticker: str | None = None,
        form_type: str | None = None,
        accession_number: str | None = None,
        top_k: int | None = None,
        mode: RetrievalMode = "hybrid",
    ) -> list[Document]:

        query = query.strip()

        if not query:
            raise ValueError("Query cannot be empty.")

        if mode != "bm25" and not query_embedding:
            raise ValueError("Query embedding cannot be empty.")

        result_limit = top_k or self.top_k

        if mode not in {
            "dense",
            "lexical",
            "hybrid",
            "bm25",
            "bm25_hybrid",
        }:
            raise ValueError(f"Unsupported retrieval mode: {mode!r}.")

        set_span_attributes(
            {
                "rag.retrieval.query_length": len(query),
                "rag.retrieval.top_k": result_limit,
                "rag.retrieval.mode": mode,
                "rag.retrieval.dense_top_k": self.dense_top_k,
                "rag.retrieval.lexical_top_k": self.lexical_top_k,
                "rag.retrieval.embedding_dimension": (
                    len(query_embedding) if query_embedding else 0
                ),
                "rag.retrieval.ticker": ticker,
                "rag.retrieval.form_type": form_type,
                "rag.retrieval.accession_number": accession_number,
                "rag.embedding.provider": (self.embeddings.__class__.__name__),
            }
        )
        set_span_input(
            {
                "query_length": len(query),
                "embedding_dimension": len(query_embedding) if query_embedding else 0,
                "ticker": ticker,
                "form_type": form_type,
                "accession_number": accession_number,
                "top_k": result_limit,
                "mode": mode,
            }
        )

        if mode != "bm25" and self.vector_store is None:
            raise RuntimeError(
                "Retriever is not ready. Run the application warmup first."
            )

        filters = self._build_filters(
            ticker=ticker,
            form_type=form_type,
            accession_number=accession_number,
            embedding_provider=self.embedding_provider,
            embedding_model=self.embedding_model,
            embedding_dimension=self.embedding_dimension,
        )

        if mode in {"bm25", "bm25_hybrid"}:
            bm25_documents = await self.bm25_store.search(
                query,
                ticker=ticker,
                form_type=form_type,
                accession_number=accession_number,
                top_k=self.lexical_top_k,
            )

            if mode == "bm25":
                documents = bm25_documents[:result_limit]
            else:
                if query_embedding is None:  # guarded above; narrows the type.
                    raise RuntimeError("Dense retrieval requires a query embedding.")
                dense_documents = await self._search_dense(
                    query_embedding,
                    filters=filters,
                    top_k=self.dense_top_k,
                )
                documents = reciprocal_rank_fuse_documents(
                    dense_documents,
                    bm25_documents,
                    top_k=result_limit,
                )
        else:
            if query_embedding is None:  # guarded above; narrows the type.
                raise RuntimeError("Vector retrieval requires a query embedding.")
            hybrid_config = None

            if mode != "dense":
                fusion_function: Any = (
                    reciprocal_rank_fusion if mode == "hybrid" else lexical_only_ranking
                )
                hybrid_config = HybridSearchConfig(
                    tsv_lang="pg_catalog.english",
                    fts_query=query,
                    fusion_function=fusion_function,
                    primary_top_k=self.dense_top_k,
                    secondary_top_k=self.lexical_top_k,
                    fusion_function_parameters=(
                        {"rrf_k": 60} if mode == "hybrid" else {}
                    ),
                )

            vector_store = self.vector_store
            if vector_store is None:
                raise RuntimeError("Retriever is not initialized.")
            documents = await vector_store.asimilarity_search_by_vector(
                embedding=query_embedding,
                k=result_limit,
                filter=filters,
                hybrid_search_config=hybrid_config,
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
            "retrieval_completed",
            query_length=len(query),
            result_count=len(documents),
            mode=mode,
            ticker=ticker,
            form_type=form_type,
            accession_number=accession_number,
        )

        return documents

    async def _search_dense(
        self,
        query_embedding: list[float],
        *,
        filters: dict,
        top_k: int,
    ) -> list[Document]:
        if self.vector_store is None:
            raise RuntimeError("Retriever is not initialized.")
        return await self.vector_store.asimilarity_search_by_vector(
            embedding=query_embedding,
            k=top_k,
            filter=filters,
        )

    @staticmethod
    def _build_filters(
        *,
        ticker: str | None,
        form_type: str | None,
        accession_number: str | None,
        embedding_provider: str,
        embedding_model: str,
        embedding_dimension: int,
    ) -> dict[str, object]:

        filters: list[dict[str, object]] = [
            {"embedding_provider": embedding_provider},
            {"embedding_model": embedding_model},
            {"embedding_dimension": embedding_dimension},
        ]

        if ticker:
            filters.append({"ticker": ticker.upper()})

        if form_type:
            filters.append({"form_type": form_type})

        if accession_number:
            filters.append({"accession_number": accession_number})

        if len(filters) == 1:
            return filters[0]

        return {"$and": filters}

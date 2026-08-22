import asyncio

import pytest
from langchain_core.documents import Document

from rag_sec import observability
from rag_sec.retrieval import Retriever


class UnusedEmbeddingModel:
    async def aembed_query(self, _query: str) -> list[float]:
        raise AssertionError("The retriever must not embed the query.")


class StubVectorStore:
    def __init__(self) -> None:
        self.options = None

    async def asimilarity_search_by_vector(self, **options):
        self.options = options
        return [Document(page_content="result")]


def build_retriever(vector_store):
    retriever = Retriever.__new__(Retriever)
    retriever.embeddings = UnusedEmbeddingModel()
    retriever.top_k = 5
    retriever.dense_top_k = 20
    retriever.lexical_top_k = 20
    retriever.vector_store = vector_store
    return retriever


def test_search_uses_precomputed_embedding_and_keeps_lexical_query(
    monkeypatch,
):
    vector_store = StubVectorStore()
    retriever = build_retriever(vector_store)

    monkeypatch.setattr(
        observability,
        "configure_observability",
        lambda: None,
    )

    documents = asyncio.run(
        retriever.search(
            "  foreign exchange  ",
            query_embedding=[0.1, 0.2, 0.3],
            ticker="AAPL",
            form_type="10-K",
        )
    )

    assert len(documents) == 1
    assert vector_store.options["embedding"] == [0.1, 0.2, 0.3]
    assert vector_store.options["hybrid_search_config"].fts_query == "foreign exchange"


def test_search_requires_warmup(monkeypatch):
    retriever = build_retriever(None)

    monkeypatch.setattr(
        observability,
        "configure_observability",
        lambda: None,
    )

    with pytest.raises(RuntimeError, match="warmup"):
        asyncio.run(
            retriever.search(
                "foreign exchange",
                query_embedding=[0.1, 0.2, 0.3],
            )
        )

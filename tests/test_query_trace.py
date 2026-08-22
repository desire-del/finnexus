import asyncio
from types import SimpleNamespace

from langchain_core.documents import Document
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from rag_sec import observability
from rag_sec.application import query as query_module
from rag_sec.generation.generator import RAGAnswer
from rag_sec.retrieval import Retriever


class StubEmbeddingModel:
    async def aembed_query(self, _query: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class StubVectorStore:
    async def asimilarity_search_by_vector(self, **_options):
        return [Document(page_content="result")]


class StubGenerator:
    async def generate(self, _question, _documents) -> RAGAnswer:
        return RAGAnswer(answer="answer", sources=[])


def test_embedding_and_retrieval_are_sibling_query_spans(monkeypatch):
    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    settings = SimpleNamespace(
        embedding=SimpleNamespace(
            provider=SimpleNamespace(value="test"),
            model_name="test-embedding",
            dimension=3,
        ),
        observability=SimpleNamespace(capture_content=False),
    )

    monkeypatch.setattr(
        observability,
        "configure_observability",
        lambda: tracer_provider,
    )
    monkeypatch.setattr(
        observability.trace,
        "get_tracer",
        tracer_provider.get_tracer,
    )
    monkeypatch.setattr(query_module, "get_settings", lambda: settings)
    retriever = Retriever.__new__(Retriever)
    retriever.embeddings = StubEmbeddingModel()
    retriever.top_k = 5
    retriever.dense_top_k = 20
    retriever.lexical_top_k = 20
    retriever.vector_store = StubVectorStore()
    runtime = SimpleNamespace(
        embedding_model=StubEmbeddingModel(),
        retriever=retriever,
        generator=StubGenerator(),
    )

    asyncio.run(
        query_module.answer_query(
            runtime,
            "foreign exchange",
            ticker="AAPL",
            form_type="10-K",
        )
    )

    spans = {span.name: span for span in exporter.get_finished_spans()}
    root_span_id = spans["rag.query"].context.span_id

    assert spans["query.embedding"].parent.span_id == root_span_id
    assert spans["retrieval.search"].parent.span_id == root_span_id

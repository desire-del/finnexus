from langchain_core.embeddings import Embeddings
from openinference.semconv.trace import OpenInferenceSpanKindValues

from rag_sec.application.runtime import RAGRuntime
from rag_sec.config import get_settings
from rag_sec.generation.generator import RAGAnswer
from rag_sec.observability import (
    Phase,
    set_span_attributes,
    set_span_input,
    set_span_output,
    track,
)


@track(
    name="query.embedding",
    phase=Phase.QUERY,
    tags=["component:embedding"],
    span_kind=OpenInferenceSpanKindValues.EMBEDDING,
)
async def embed_query(
    query: str,
    *,
    model: Embeddings,
) -> list[float]:
    """Embed and validate a query before retrieval."""
    query = query.strip()

    if not query:
        raise ValueError("Query cannot be empty.")

    settings = get_settings()
    embedding_settings = settings.embedding
    span_input: dict[str, object] = {
        "query_length": len(query),
    }

    if settings.observability.capture_content:
        span_input["query"] = query

    set_span_input(span_input)
    set_span_attributes(
        {
            "rag.embedding.provider": embedding_settings.provider.value,
            "rag.embedding.model": embedding_settings.model_name,
            "rag.embedding.expected_dimension": (embedding_settings.dimension),
        }
    )

    vector = await model.aembed_query(query)
    actual_dimension = len(vector)

    if actual_dimension != embedding_settings.dimension:
        raise ValueError(
            "Query embedding dimension does not match the configured "
            f"dimension: expected {embedding_settings.dimension}, "
            f"received {actual_dimension}."
        )

    set_span_attributes({"rag.embedding.dimension": actual_dimension})
    set_span_output({"dimension": actual_dimension})
    return vector


@track(
    name="rag.query",
    phase=Phase.QUERY,
    tags=["workflow:rag"],
    span_kind=OpenInferenceSpanKindValues.AGENT,
)
async def answer_query(
    runtime: RAGRuntime,
    question: str,
    *,
    ticker: str,
    form_type: str,
) -> RAGAnswer:
    settings = get_settings()

    set_span_attributes(
        {
            "rag.question.length": len(question),
            "rag.company.ticker": ticker,
            "rag.filing.form_type": form_type,
        }
    )
    trace_input: dict[str, object] = {
        "ticker": ticker,
        "form_type": form_type,
    }

    if settings.observability.capture_content:
        trace_input["query"] = question
    else:
        trace_input["query_length"] = len(question)

    set_span_input(trace_input)

    query_embedding = await embed_query(
        question,
        model=runtime.embedding_model,
    )

    documents = await runtime.retriever.search(
        question,
        query_embedding=query_embedding,
        ticker=ticker,
        form_type=form_type,
    )

    result = await runtime.generator.generate(
        question,
        documents,
    )

    trace_output: dict[str, object] = {
        "retrieved_documents": len(documents),
        "cited_sources": len(result.sources),
    }

    if settings.observability.capture_content:
        trace_output["response"] = result.answer
        trace_output["sources"] = [
            source.model_dump(exclude_none=True) for source in result.sources
        ]
    else:
        trace_output["answer_length"] = len(result.answer)

    set_span_output(trace_output)
    return result

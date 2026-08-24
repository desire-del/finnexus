from langchain_core.documents import Document

from rag_sec.application.query import execute_query, retrieve_query
from rag_sec.application.runtime import RAGRuntime
from rag_sec.evaluation.models import (
    EvaluationCase,
    EvaluationRun,
    EvaluationRunMetrics,
    RetrievedEvidence,
)
from rag_sec.retrieval.retriever import RetrievalMode


def to_evidence(document: Document, rank: int) -> RetrievedEvidence:
    metadata = document.metadata
    return RetrievedEvidence(
        text=document.page_content,
        rank=rank,
        chunk_id=metadata.get("chunk_id"),
        accession_number=metadata.get("accession_number"),
        section=metadata.get("section"),
        item=metadata.get("item"),
        page=metadata.get("page"),
        metadata=metadata,
    )


async def run_case(
    runtime: RAGRuntime,
    case: EvaluationCase,
    *,
    generate: bool = True,
    mode: RetrievalMode | None = None,
    top_k: int | None = None,
) -> EvaluationRun:
    try:
        if not generate:
            if not case.accession_number:
                raise ValueError("Missing accession number.")
            retrieval = await retrieve_query(
                runtime,
                case.question,
                accession_number=case.accession_number,
                top_k=top_k,
                mode=mode,
            )
            return EvaluationRun(
                case_id=case.id,
                retrieved_evidence=[
                    to_evidence(document, rank)
                    for rank, document in enumerate(retrieval.documents, start=1)
                ],
                metrics=EvaluationRunMetrics(
                    total_latency_ms=(
                        retrieval.embedding_latency_ms
                        + retrieval.retrieval_latency_ms
                    ),
                    embedding_latency_ms=retrieval.embedding_latency_ms,
                    retrieval_latency_ms=retrieval.retrieval_latency_ms,
                ),
            )

        execution = await execute_query(
            runtime=runtime,
            question=case.question,
            ticker=case.ticker,
            form_type=case.form_type,
        )

        answer = execution.answer

        retrieved_evidence = [
            to_evidence(document, rank)
            for rank, document in enumerate(
                execution.documents,
                start=1,
            )
        ]

        cited_source_ids = [source.source_id for source in answer.sources]

        return EvaluationRun(
            case_id=case.id,
            generated_answer=answer.answer,
            retrieved_evidence=retrieved_evidence,
            cited_source_ids=cited_source_ids,
            abstained=None,
            metrics=EvaluationRunMetrics(
                total_latency_ms=answer.metrics.total_latency_ms,
                embedding_latency_ms=answer.metrics.embedding_latency_ms,
                retrieval_latency_ms=answer.metrics.retrieval_latency_ms,
                generation_latency_ms=answer.metrics.generation_latency_ms,
                input_tokens=answer.usage.input_tokens,
                output_tokens=answer.usage.output_tokens,
            ),
        )

    except Exception as exc:  # noqa: BLE001 - isolate failures per evaluation case.
        return EvaluationRun(
            case_id=case.id,
            error=str(exc),
        )


async def run_dataset(
    runtime: RAGRuntime,
    cases: list[EvaluationCase],
    *,
    generate: bool = True,
    mode: RetrievalMode | None = None,
    top_k: int | None = None,
) -> list[EvaluationRun]:
    return [
        await run_case(
            runtime=runtime,
            case=case,
            generate=generate,
            mode=mode,
            top_k=top_k,
        )
        for case in cases
    ]

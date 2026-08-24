from rag_sec.application.query import execute_query
from rag_sec.application.runtime import RAGRuntime
from rag_sec.evaluation.models import (
    EvaluationCase,
    EvaluationRun,
    EvaluationRunMetrics,
)
from rag_sec.evaluation.retrieval import to_evidence


async def run_case(
    runtime: RAGRuntime,
    case: EvaluationCase,
) -> EvaluationRun:
    try:
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
) -> list[EvaluationRun]:
    results = []

    for case in cases:
        result = await run_case(
            runtime=runtime,
            case=case,
        )
        results.append(result)

    return results

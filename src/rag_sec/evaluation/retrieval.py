from collections import defaultdict
from dataclasses import dataclass
from statistics import mean

from rag_sec.application.runtime import RAGRuntime
from rag_sec.evaluation.evaluators.retrieval import evaluate_retrieval
from rag_sec.evaluation.models import (
    EvaluationCase,
    EvaluationRun,
)
from rag_sec.evaluation.runner import run_dataset
from rag_sec.retrieval.retriever import RetrievalMode


@dataclass(frozen=True)
class RetrievalResult:
    metrics: dict[str, float]
    records: list[dict]
    runs: list[EvaluationRun]


class RetrievalEvaluator:
    """Run one retrieval configuration and produce its complete result."""

    def __init__(self, runtime: RAGRuntime) -> None:
        self.runtime = runtime

    async def evaluate(
        self,
        cases: list[EvaluationCase],
        *,
        mode: RetrievalMode | None = None,
        top_k: int,
        ks: tuple[int, ...],
    ) -> RetrievalResult:
        runs = await run_dataset(
            self.runtime,
            cases,
            generate=False,
            mode=mode,
            top_k=top_k,
        )
        metrics, records = aggregate_runs(cases, runs, ks=ks)
        return RetrievalResult(metrics=metrics, records=records, runs=runs)


def aggregate_runs(
    cases: list[EvaluationCase],
    runs: list[EvaluationRun],
    *,
    ks: tuple[int, ...],
) -> tuple[dict[str, float], list[dict]]:
    values: dict[str, list[float]] = defaultdict(list)
    records: list[dict] = []
    for case, run in zip(cases, runs, strict=True):
        scores = evaluate_retrieval(case, run, ks=ks)
        score_values = {score.metric: score.score for score in scores}
        for score in scores:
            if score.score is not None:
                values[score.metric].append(score.score)
        records.append(
            {
                "case_id": case.id,
                "question": case.question,
                "accession_number": case.accession_number,
                "scores": score_values,
                "retrieved_evidence": [
                    evidence.model_dump(mode="json")
                    for evidence in run.retrieved_evidence
                ],
                "metrics": run.metrics.model_dump(mode="json"),
                "error": run.error,
            }
        )
    metrics = {
        (
            name.replace("reciprocal_rank@", "mrr@")
            if name.startswith("reciprocal_rank@")
            else name
        ): mean(items)
        for name, items in values.items()
    }
    return metrics, records

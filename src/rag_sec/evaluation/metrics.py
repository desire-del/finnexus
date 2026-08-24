from collections.abc import Callable, Sequence
from dataclasses import dataclass

from rag_sec.evaluation.evaluators.matching import EvidenceMatchConfig
from rag_sec.evaluation.evaluators.retrieval import (
    hit_at_k,
    recall_at_k,
    reciprocal_rank,
)
from rag_sec.evaluation.models import EvaluationCase, EvaluationRun

MetricFunction = Callable[[EvaluationCase, EvaluationRun], float | None]


@dataclass(frozen=True)
class EvaluationMetric:
    """A named callable and the name used for its aggregate mean."""

    name: str
    function: MetricFunction
    aggregate_name: str | None = None

    def __call__(self, case: EvaluationCase, run: EvaluationRun) -> float | None:
        return self.function(case, run)

    @property
    def summary_name(self) -> str:
        return self.aggregate_name or self.name


Metric = EvaluationMetric | MetricFunction


def _requires_references(function: MetricFunction) -> MetricFunction:
    def metric(case: EvaluationCase, run: EvaluationRun) -> float | None:
        if not case.reference_evidence:
            return None
        return function(case, run)

    return metric


def _hit_metric(k: int, config: EvidenceMatchConfig) -> MetricFunction:
    def metric(case: EvaluationCase, run: EvaluationRun) -> float:
        return hit_at_k(case, run, k, config=config)

    return _requires_references(metric)


def _recall_metric(k: int, config: EvidenceMatchConfig) -> MetricFunction:
    def metric(case: EvaluationCase, run: EvaluationRun) -> float:
        return recall_at_k(case, run, k, config=config)

    return _requires_references(metric)


def _reciprocal_rank_metric(k: int, config: EvidenceMatchConfig) -> MetricFunction:
    def metric(case: EvaluationCase, run: EvaluationRun) -> float:
        return reciprocal_rank(case, run, k, config=config)

    return _requires_references(metric)


def retrieval_metrics(
    ks: Sequence[int] = (1, 3, 5, 10, 20),
) -> tuple[EvaluationMetric, ...]:
    """Build deterministic retrieval metrics with the frozen evidence matcher."""
    cutoffs = tuple(dict.fromkeys(ks))
    if not cutoffs or any(k <= 0 for k in cutoffs):
        raise ValueError("Metric cutoffs must contain positive integers.")

    config = EvidenceMatchConfig()
    metrics: list[EvaluationMetric] = []
    for k in cutoffs:
        metrics.extend(
            (
                EvaluationMetric(
                    name=f"hit@{k}",
                    function=_hit_metric(k, config),
                ),
                EvaluationMetric(
                    name=f"recall@{k}",
                    function=_recall_metric(k, config),
                ),
                EvaluationMetric(
                    name=f"reciprocal_rank@{k}",
                    aggregate_name=f"mrr@{k}",
                    function=_reciprocal_rank_metric(k, config),
                ),
            )
        )
    return tuple(metrics)

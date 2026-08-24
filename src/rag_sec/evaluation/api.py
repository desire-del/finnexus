from collections import defaultdict
from collections.abc import Sequence
from statistics import mean

from rag_sec.application.runtime import RAGRuntime
from rag_sec.config import RetrievalSettings
from rag_sec.evaluation.metrics import EvaluationMetric, Metric
from rag_sec.evaluation.models import EvaluationCase, EvaluationScore
from rag_sec.evaluation.result import CaseEvaluationResult, EvaluationResult
from rag_sec.evaluation.runner import run_dataset


def _metric_name(metric: Metric) -> str:
    if isinstance(metric, EvaluationMetric):
        return metric.name
    return getattr(metric, "__name__", metric.__class__.__name__)


def _summary_name(metric: Metric) -> str:
    return (
        metric.summary_name
        if isinstance(metric, EvaluationMetric)
        else _metric_name(metric)
    )


async def evaluate(
    *,
    dataset: Sequence[EvaluationCase],
    settings: RetrievalSettings,
    metrics: Sequence[Metric],
    runtime: RAGRuntime | None = None,
) -> EvaluationResult:
    """Evaluate the configured production retrieval pipeline without persistence."""
    selected_metrics = tuple(metrics)
    if not selected_metrics:
        raise ValueError("At least one evaluation metric is required.")

    names = [_metric_name(metric) for metric in selected_metrics]
    if len(names) != len(set(names)):
        raise ValueError("Evaluation metric names must be unique.")

    validate_corpus = getattr(dataset, "validate_corpus", None)
    if validate_corpus is not None:
        await validate_corpus()

    owns_runtime = runtime is None
    active_runtime = runtime or RAGRuntime()
    try:
        active_runtime.retriever.settings = settings
        await active_runtime.warmup_retrieval()
        cases = list(dataset)
        runs = await run_dataset(
            active_runtime,
            cases,
            generate=False,
            top_k=settings.top_k,
        )

        aggregate_values: dict[str, list[float]] = defaultdict(list)
        case_results: list[CaseEvaluationResult] = []
        for case, run in zip(cases, runs, strict=True):
            scores = []
            for metric in selected_metrics:
                name = _metric_name(metric)
                if run.error:
                    score = EvaluationScore(
                        evaluator="evaluation",
                        metric=name,
                        score=None,
                        label="error",
                        explanation=run.error,
                        evaluator_type="deterministic",
                    )
                else:
                    try:
                        value = metric(case, run)
                    except Exception as exc:  # noqa: BLE001 - custom metric boundary.
                        score = EvaluationScore(
                            evaluator="evaluation",
                            metric=name,
                            score=None,
                            label="error",
                            explanation=str(exc),
                            evaluator_type="deterministic",
                        )
                    else:
                        score = EvaluationScore(
                            evaluator="evaluation",
                            metric=name,
                            score=value,
                            label="not_applicable" if value is None else None,
                            evaluator_type="deterministic",
                        )
                        if value is not None:
                            aggregate_values[_summary_name(metric)].append(value)
                scores.append(score)
            case_results.append(
                CaseEvaluationResult(case=case, run=run, scores=tuple(scores))
            )

        dataset_metadata = dict(getattr(dataset, "metadata", {}))
        return EvaluationResult(
            dataset_metadata=dataset_metadata,
            settings=settings,
            aggregate_metrics={
                name: mean(values) for name, values in aggregate_values.items()
            },
            cases=tuple(case_results),
        )
    finally:
        if owns_runtime:
            await active_runtime.shutdown()

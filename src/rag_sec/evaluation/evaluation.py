from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import mean
from typing import TYPE_CHECKING, Any

from langchain_core.documents import Document

from rag_sec.application.query import retrieve_query
from rag_sec.application.runtime import RAGRuntime
from rag_sec.config import RetrievalSettings
from rag_sec.evaluation.metrics import EvaluationMetric, Metric
from rag_sec.evaluation.models import (
    EvaluationCase,
    EvaluationRun,
    EvaluationRunMetrics,
    EvaluationScore,
    RetrievedEvidence,
)

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True)
class CaseEvaluationResult:
    """One normalized case, its pipeline output, and its metric scores."""

    case: EvaluationCase
    run: EvaluationRun
    scores: tuple[EvaluationScore, ...]

    @property
    def metric_values(self) -> dict[str, float | None]:
        return {score.metric: score.score for score in self.scores}

    def to_record(self) -> dict[str, Any]:
        """Flatten scalar fields while retaining rich objects for inspection."""
        case = self.case
        metadata = case.metadata
        return {
            "case_id": case.id,
            "dataset_name": case.dataset_name,
            "dataset_version": case.dataset_version,
            "subset_label": metadata.get("dataset_subset_label"),
            "question": case.question,
            "question_type": metadata.get("question_type"),
            "question_reasoning": metadata.get("question_reasoning"),
            "reference_answer": case.reference_answer,
            "justification": metadata.get("justification"),
            "company": case.company,
            "ticker": case.ticker,
            "cik": metadata.get("company_cik"),
            "accession_number": case.accession_number,
            "document_name": metadata.get("doc_name"),
            "form_type": case.form_type,
            "document_period": metadata.get("doc_period"),
            "gold_evidence_count": len(case.reference_evidence),
            "retrieved_evidence_count": len(self.run.retrieved_evidence),
            "total_latency_ms": self.run.metrics.total_latency_ms,
            "embedding_latency_ms": self.run.metrics.embedding_latency_ms,
            "retrieval_latency_ms": self.run.metrics.retrieval_latency_ms,
            "generation_latency_ms": self.run.metrics.generation_latency_ms,
            "input_tokens": self.run.metrics.input_tokens,
            "output_tokens": self.run.metrics.output_tokens,
            "error": self.run.error,
            **self.metric_values,
            "reference_evidence": case.reference_evidence,
            "retrieved_evidence": self.run.retrieved_evidence,
            "case": case,
            "run": self.run,
        }


@dataclass(frozen=True)
class EvaluationResult:
    """Notebook-friendly aggregate and per-case evaluation output."""

    dataset_metadata: dict[str, Any]
    settings: RetrievalSettings
    aggregate_metrics: dict[str, float]
    cases: tuple[CaseEvaluationResult, ...]

    def __len__(self) -> int:
        return len(self.cases)

    @property
    def errors(self) -> tuple[CaseEvaluationResult, ...]:
        return tuple(result for result in self.cases if result.run.error)

    def summary(self) -> dict[str, Any]:
        """Return aggregate metrics with basic execution counts."""
        return {
            "dataset": self.dataset_metadata.get("name"),
            "subset": self.dataset_metadata.get("subset"),
            "retrieval_mode": self.settings.mode,
            "hybrid_lexical_backend": (
                self.settings.hybrid_lexical_backend
                if self.settings.mode == "hybrid"
                else None
            ),
            "case_count": len(self.cases),
            "error_count": len(self.errors),
            **self.aggregate_metrics,
        }

    def to_records(self) -> list[dict[str, Any]]:
        """Return one analysis record per evaluated case."""
        settings = self.settings
        configuration = {
            "retrieval_mode": settings.mode,
            "retrieval_top_k": settings.top_k,
            "dense_candidate_k": settings.dense_candidate_k,
            "fts_candidate_k": settings.fts_candidate_k,
            "bm25_candidate_k": settings.bm25_candidate_k,
            "hybrid_lexical_backend": (
                settings.hybrid_lexical_backend if settings.mode == "hybrid" else None
            ),
            "rrf_k": settings.rrf_k if settings.mode == "hybrid" else None,
            "dense_weight": (
                settings.dense_weight if settings.mode == "hybrid" else None
            ),
            "lexical_weight": (
                settings.lexical_weight if settings.mode == "hybrid" else None
            ),
        }
        return [{**result.to_record(), **configuration} for result in self.cases]

    def to_dataframe(self) -> "pd.DataFrame":
        """Return one notebook analysis row per evaluated case."""
        import pandas as pd

        return pd.DataFrame.from_records(self.to_records())


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


def _to_evidence(document: Document, rank: int) -> RetrievedEvidence:
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


async def _evaluate_case(
    runtime: RAGRuntime,
    case: EvaluationCase,
    *,
    top_k: int,
) -> EvaluationRun:
    try:
        if not case.accession_number:
            raise ValueError("Missing accession number.")

        retrieval = await retrieve_query(
            runtime,
            case.question,
            accession_number=case.accession_number,
            top_k=top_k,
        )
        return EvaluationRun(
            case_id=case.id,
            retrieved_evidence=[
                _to_evidence(document, rank)
                for rank, document in enumerate(retrieval.documents, start=1)
            ],
            metrics=EvaluationRunMetrics(
                total_latency_ms=(
                    retrieval.embedding_latency_ms + retrieval.retrieval_latency_ms
                ),
                embedding_latency_ms=retrieval.embedding_latency_ms,
                retrieval_latency_ms=retrieval.retrieval_latency_ms,
            ),
        )
    except Exception as exc:  # noqa: BLE001 - isolate failures per evaluation case.
        return EvaluationRun(case_id=case.id, error=str(exc))


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
        runs = [
            await _evaluate_case(active_runtime, case, top_k=settings.top_k)
            for case in cases
        ]

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

        return EvaluationResult(
            dataset_metadata=dict(getattr(dataset, "metadata", {})),
            settings=settings,
            aggregate_metrics={
                name: mean(values) for name, values in aggregate_values.items()
            },
            cases=tuple(case_results),
        )
    finally:
        if owns_runtime:
            await active_runtime.shutdown()

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from rag_sec.config import RetrievalSettings
from rag_sec.evaluation.models import EvaluationCase, EvaluationRun, EvaluationScore

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

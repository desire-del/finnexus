from collections import defaultdict
from dataclasses import dataclass
from statistics import mean

from rag_sec.application.runtime import RAGRuntime
from rag_sec.evaluation.evaluators.retrieval import evaluate_retrieval
from rag_sec.evaluation.models import (
    EvaluationCase,
    EvaluationRun,
    RetrievedEvidence,
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
        mode: RetrievalMode,
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
    max_k = max(ks)
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
        (f"mrr@{max_k}" if name == f"reciprocal_rank@{max_k}" else name): mean(items)
        for name, items in values.items()
    }
    return metrics, records


def evidence_identity(evidence: RetrievedEvidence) -> str:
    return evidence.chunk_id or evidence.text


def deduplicated_union(
    *rankings: list[RetrievedEvidence],
) -> list[RetrievedEvidence]:
    candidates: list[RetrievedEvidence] = []
    seen: set[str] = set()
    for ranking in rankings:
        for evidence in ranking:
            identity = evidence_identity(evidence)
            if identity not in seen:
                seen.add(identity)
                candidates.append(evidence)
    return candidates


def weighted_rrf(
    dense_run: EvaluationRun,
    lexical_run: EvaluationRun,
    *,
    dense_depth: int,
    lexical_depth: int,
    dense_weight: float,
    lexical_weight: float,
    rrf_k: int,
    top_k: int,
) -> EvaluationRun:
    scores: dict[str, float] = {}
    evidence_by_id: dict[str, RetrievedEvidence] = {}
    for evidence_list, weight in (
        (dense_run.retrieved_evidence[:dense_depth], dense_weight),
        (lexical_run.retrieved_evidence[:lexical_depth], lexical_weight),
    ):
        for rank, evidence in enumerate(evidence_list, start=1):
            identity = evidence_identity(evidence)
            evidence_by_id.setdefault(identity, evidence)
            scores[identity] = scores.get(identity, 0) + weight / (rrf_k + rank)

    identities = sorted(scores, key=lambda identity: scores[identity], reverse=True)[
        :top_k
    ]
    return EvaluationRun(
        case_id=dense_run.case_id,
        retrieved_evidence=[
            evidence_by_id[identity].model_copy(update={"rank": rank})
            for rank, identity in enumerate(identities, start=1)
        ],
    )


def contribution_diagnostics(
    dense: RetrievalResult,
    lexical: RetrievalResult,
    hybrid: RetrievalResult,
    *,
    candidate_k: int = 20,
    final_k: int = 5,
    lexical_label: str = "lexical",
) -> dict[str, int]:
    def successes(records: list[dict], k: int) -> set[str]:
        return {
            record["case_id"] for record in records if record["scores"][f"hit@{k}"] == 1
        }

    def first_rank(record: dict) -> int | None:
        reciprocal_rank = record["scores"][f"reciprocal_rank@{candidate_k}"]
        return round(1 / reciprocal_rank) if reciprocal_rank else None

    dense_candidate = successes(dense.records, candidate_k)
    lexical_candidate = successes(lexical.records, candidate_k)
    hybrid_candidate = successes(hybrid.records, candidate_k)
    dense_final = successes(dense.records, final_k)
    hybrid_final = successes(hybrid.records, final_k)
    rank_improved = rank_degraded = recall_improved = recall_degraded = 0

    for dense_record, hybrid_record in zip(dense.records, hybrid.records, strict=True):
        dense_rank = first_rank(dense_record)
        hybrid_rank = first_rank(hybrid_record)
        rank_improved += hybrid_rank is not None and (
            dense_rank is None or hybrid_rank < dense_rank
        )
        rank_degraded += dense_rank is not None and (
            hybrid_rank is None or hybrid_rank > dense_rank
        )
        dense_recall = dense_record["scores"][f"recall@{final_k}"]
        hybrid_recall = hybrid_record["scores"][f"recall@{final_k}"]
        recall_improved += hybrid_recall > dense_recall
        recall_degraded += hybrid_recall < dense_recall

    return {
        f"dense_successes@{candidate_k}": len(dense_candidate),
        f"{lexical_label}_successes@{candidate_k}": len(lexical_candidate),
        f"hybrid_successes@{candidate_k}": len(hybrid_candidate),
        f"{lexical_label}_only_successes@{candidate_k}": len(
            lexical_candidate - dense_candidate
        ),
        f"dense_only_successes@{candidate_k}": len(dense_candidate - lexical_candidate),
        f"fused_only_vs_dense@{candidate_k}": len(hybrid_candidate - dense_candidate),
        f"dense_lost_by_fusion@{candidate_k}": len(dense_candidate - hybrid_candidate),
        f"dense_miss_to_hybrid_success@{final_k}": len(hybrid_final - dense_final),
        f"dense_success_to_hybrid_miss@{final_k}": len(dense_final - hybrid_final),
        "rank_improved": rank_improved,
        "rank_degraded": rank_degraded,
        f"recall_improved@{final_k}": recall_improved,
        f"recall_degraded@{final_k}": recall_degraded,
    }

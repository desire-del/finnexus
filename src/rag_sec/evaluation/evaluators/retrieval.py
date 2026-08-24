from rag_sec.evaluation.evaluators.matching import (
    EvidenceMatchConfig,
    evidence_matches,
)
from rag_sec.evaluation.models import (
    EvaluationCase,
    EvaluationRun,
    EvaluationScore,
    RetrievedEvidence,
)


def is_relevant(
    case: EvaluationCase,
    evidence: RetrievedEvidence,
    *,
    config: EvidenceMatchConfig,
) -> bool:
    return any(
        evidence_matches(
            reference,
            evidence,
            config=config,
        )
        for reference in case.reference_evidence
    )


def hit_at_k(
    case: EvaluationCase,
    run: EvaluationRun,
    k: int,
    *,
    config: EvidenceMatchConfig,
) -> float:
    top_k = sorted(
        run.retrieved_evidence,
        key=lambda evidence: evidence.rank,
    )[:k]

    return float(
        any(
            is_relevant(
                case,
                evidence,
                config=config,
            )
            for evidence in top_k
        )
    )


def reciprocal_rank(
    case: EvaluationCase,
    run: EvaluationRun,
    *,
    config: EvidenceMatchConfig,
) -> float:
    evidence = sorted(
        run.retrieved_evidence,
        key=lambda evidence: evidence.rank,
    )

    for item in evidence:
        if is_relevant(
            case,
            item,
            config=config,
        ):
            return 1.0 / item.rank

    return 0.0


def recall_at_k(
    case: EvaluationCase,
    run: EvaluationRun,
    k: int,
    *,
    config: EvidenceMatchConfig,
) -> float:
    references = case.reference_evidence

    if not references:
        return 0.0

    top_k = sorted(
        run.retrieved_evidence,
        key=lambda evidence: evidence.rank,
    )[:k]

    matched_references = 0

    for reference in references:
        matched = any(
            evidence_matches(
                reference,
                evidence,
                config=config,
            )
            for evidence in top_k
        )

        if matched:
            matched_references += 1

    return matched_references / len(references)


def evaluate_retrieval(
    case: EvaluationCase,
    run: EvaluationRun,
    *,
    ks: tuple[int, ...] = (1, 3, 5),
    config: EvidenceMatchConfig | None = None,
) -> list[EvaluationScore]:
    config = config or EvidenceMatchConfig()

    if not case.reference_evidence:
        return [
            EvaluationScore(
                evaluator="retrieval",
                metric="retrieval",
                score=None,
                label="not_applicable",
                explanation=("The evaluation case has no reference evidence."),
                evaluator_type="deterministic",
            )
        ]

    if run.error:
        return [
            EvaluationScore(
                evaluator="retrieval",
                metric="retrieval",
                score=None,
                label="error",
                explanation=run.error,
                evaluator_type="deterministic",
            )
        ]

    scores: list[EvaluationScore] = []

    for k in ks:
        scores.append(
            EvaluationScore(
                evaluator="retrieval",
                metric=f"hit@{k}",
                score=hit_at_k(
                    case,
                    run,
                    k,
                    config=config,
                ),
                evaluator_type="deterministic",
            )
        )

        scores.append(
            EvaluationScore(
                evaluator="retrieval",
                metric=f"recall@{k}",
                score=recall_at_k(
                    case,
                    run,
                    k,
                    config=config,
                ),
                evaluator_type="deterministic",
            )
        )

    scores.append(
        EvaluationScore(
            evaluator="retrieval",
            metric=f"reciprocal_rank@{max(ks)}",
            score=reciprocal_rank(
                case,
                run,
                config=config,
            ),
            evaluator_type="deterministic",
        )
    )

    return scores

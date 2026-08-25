import re
from collections import Counter
from dataclasses import dataclass

from rag_sec.evaluation.models import ReferenceEvidence, RetrievedEvidence


@dataclass(frozen=True)
class EvidenceMatchConfig:
    min_token_recall: float = 0.80
    min_local_token_recall: float = 0.85
    local_window_tokens: int = 24
    local_window_stride: int = 8
    min_tokens: int = 8


def normalize_text(text: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", text.casefold())
    return " ".join(tokens)


def token_recall(reference: str, candidate: str) -> float:
    reference_tokens = reference.split()
    candidate_tokens = candidate.split()

    if not reference_tokens:
        return 0.0

    reference_counts = Counter(reference_tokens)
    candidate_counts = Counter(candidate_tokens)
    matched = sum((reference_counts & candidate_counts).values())
    return matched / len(reference_tokens)


def local_token_recall(
    reference: str,
    candidate: str,
    *,
    window_size: int,
    stride: int,
) -> float:
    """Return the best coverage of a local reference window."""
    if window_size <= 0:
        raise ValueError("Local window size must be positive.")
    if stride <= 0:
        raise ValueError("Local window stride must be positive.")

    reference_tokens = reference.split()
    if not reference_tokens:
        return 0.0
    if len(reference_tokens) <= window_size:
        return token_recall(reference, candidate)

    best_score = 0.0
    for start in range(0, len(reference_tokens), stride):
        window = reference_tokens[start : start + window_size]
        if len(window) < window_size // 2:
            break

        best_score = max(best_score, token_recall(" ".join(window), candidate))
        if best_score == 1.0:
            break

    return best_score


def evidence_matches(
    reference: ReferenceEvidence,
    retrieved: RetrievedEvidence,
    *,
    config: EvidenceMatchConfig | None = None,
) -> bool:
    config = config or EvidenceMatchConfig()

    if (
        reference.chunk_id
        and retrieved.chunk_id
        and reference.chunk_id == retrieved.chunk_id
    ):
        return True

    if (
        reference.accession_number
        and retrieved.accession_number
        and reference.accession_number != retrieved.accession_number
    ):
        return False

    if not reference.text:
        return False

    reference_text = normalize_text(reference.text)
    retrieved_text = normalize_text(retrieved.text)
    if not reference_text or not retrieved_text:
        return False

    reference_tokens = reference_text.split()
    retrieved_tokens = retrieved_text.split()
    if len(reference_tokens) >= config.min_tokens and reference_text in retrieved_text:
        return True
    if len(retrieved_tokens) >= config.min_tokens and retrieved_text in reference_text:
        return True
    if token_recall(reference_text, retrieved_text) >= config.min_token_recall:
        return True

    return (
        local_token_recall(
            reference_text,
            retrieved_text,
            window_size=config.local_window_tokens,
            stride=config.local_window_stride,
        )
        >= config.min_local_token_recall
    )

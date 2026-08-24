from collections.abc import Sequence

from langchain_core.documents import Document


def weighted_reciprocal_rank_fusion(
    dense_documents: Sequence[Document],
    lexical_documents: Sequence[Document],
    *,
    dense_weight: float,
    lexical_weight: float,
    rrf_k: int,
    top_k: int,
) -> list[Document]:
    """Fuse dense and lexical rankings using stable chunk identities."""
    scores: dict[str, float] = {}
    documents: dict[str, Document] = {}
    ranks: dict[str, dict[str, int]] = {}

    for branch, ranked_documents, weight in (
        ("dense", dense_documents, dense_weight),
        ("lexical", lexical_documents, lexical_weight),
    ):
        for rank, document in enumerate(ranked_documents, start=1):
            identity = str(
                document.metadata.get("chunk_id")
                or document.metadata.get("id")
                or document.page_content
            )
            documents.setdefault(identity, document)
            ranks.setdefault(identity, {})[branch] = rank
            scores[identity] = scores.get(identity, 0.0) + weight / (rrf_k + rank)

    identities = sorted(scores, key=lambda identity: scores[identity], reverse=True)
    return [
        documents[identity].model_copy(
            update={
                "metadata": {
                    **documents[identity].metadata,
                    "rrf_score": scores[identity],
                    "dense_rank": ranks[identity].get("dense"),
                    "lexical_rank": ranks[identity].get("lexical"),
                }
            }
        )
        for identity in identities[:top_k]
    ]

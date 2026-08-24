from collections.abc import Sequence

from langchain_core.documents import Document


def reciprocal_rank_fuse_documents(
    dense_documents: Sequence[Document],
    lexical_documents: Sequence[Document],
    *,
    top_k: int,
    rrf_k: int = 60,
) -> list[Document]:
    """Fuse two ranked lists while preserving document metadata."""
    scores: dict[str, float] = {}
    documents: dict[str, Document] = {}

    for ranked_documents in (dense_documents, lexical_documents):
        for rank, document in enumerate(ranked_documents, start=1):
            identity = str(
                document.metadata.get("chunk_id")
                or document.metadata.get("id")
                or document.page_content
            )
            documents.setdefault(identity, document)
            scores[identity] = scores.get(identity, 0.0) + 1 / (rrf_k + rank)

    identities = sorted(scores, key=lambda identity: scores[identity], reverse=True)
    return [documents[identity] for identity in identities[:top_k]]

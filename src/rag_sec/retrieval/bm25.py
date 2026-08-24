import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from langchain_core.documents import Document

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[.-][a-z0-9]+)*")


@dataclass(frozen=True)
class BM25Result:
    document: Document
    score: float


def tokenize(text: str) -> list[str]:
    """Tokenize financial text while retaining numbers and compound terms."""
    return TOKEN_PATTERN.findall(text.casefold())


def rank_bm25(
    query: str,
    documents: Sequence[Document],
    *,
    top_k: int,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[BM25Result]:
    """Rank documents with Okapi BM25 using the filtered corpus as context."""
    if top_k <= 0 or not documents:
        return []

    query_terms = tokenize(query)
    if not query_terms:
        return []

    tokenized_documents = [tokenize(document.page_content) for document in documents]
    document_count = len(tokenized_documents)
    average_length = sum(map(len, tokenized_documents)) / document_count
    if average_length == 0:
        return []

    document_frequencies: Counter[str] = Counter()
    for tokens in tokenized_documents:
        document_frequencies.update(set(tokens))

    query_frequencies = Counter(query_terms)
    scored: list[BM25Result] = []

    for document, tokens in zip(documents, tokenized_documents, strict=True):
        if not tokens:
            continue

        term_frequencies = Counter(tokens)
        length_normalization = 1 - b + b * len(tokens) / average_length
        score = 0.0

        for term, query_frequency in query_frequencies.items():
            term_frequency = term_frequencies.get(term, 0)
            if term_frequency == 0:
                continue

            frequency = document_frequencies[term]
            inverse_document_frequency = math.log(
                1 + (document_count - frequency + 0.5) / (frequency + 0.5)
            )
            score += (
                query_frequency
                * inverse_document_frequency
                * (
                    term_frequency
                    * (k1 + 1)
                    / (term_frequency + k1 * length_normalization)
                )
            )

        if score > 0:
            scored.append(BM25Result(document=document, score=score))

    scored.sort(key=lambda result: result.score, reverse=True)
    return scored[:top_k]


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

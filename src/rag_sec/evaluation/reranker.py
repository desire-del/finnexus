import asyncio
from functools import cached_property

from sentence_transformers import CrossEncoder

from rag_sec.config import RerankerSettings
from rag_sec.evaluation.models import RetrievedEvidence


class CrossEncoderReranker:
    """Lazy local cross-encoder used only by evaluation experiments."""

    def __init__(self, settings: RerankerSettings):
        self.settings = settings

    @cached_property
    def model(self) -> CrossEncoder:
        return CrossEncoder(
            self.settings.model_name,
            max_length=self.settings.max_length,
        )

    def warmup(self) -> None:
        self.model.predict(
            [("warmup query", "warmup document")],
            batch_size=1,
            show_progress_bar=False,
        )

    async def rerank(
        self,
        query: str,
        candidates: list[RetrievedEvidence],
        *,
        top_k: int,
    ) -> list[RetrievedEvidence]:
        return await asyncio.to_thread(
            self._rerank_sync,
            query,
            candidates,
            top_k,
        )

    def _rerank_sync(
        self,
        query: str,
        candidates: list[RetrievedEvidence],
        top_k: int,
    ) -> list[RetrievedEvidence]:
        if not candidates or top_k <= 0:
            return []

        scores = self.model.predict(
            [(query, candidate.text) for candidate in candidates],
            batch_size=self.settings.batch_size,
            show_progress_bar=False,
        )
        ranked = sorted(
            zip(candidates, scores, strict=True),
            key=lambda item: float(item[1]),
            reverse=True,
        )[:top_k]
        return [
            candidate.model_copy(update={"rank": rank, "score": float(score)})
            for rank, (candidate, score) in enumerate(ranked, start=1)
        ]

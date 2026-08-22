from openai import AsyncOpenAI, APIError

from rag_sec.config import get_settings
from rag_sec.exceptions import EmbeddingError
from rag_sec.logging import get_logger

from rag_sec.schemas.chunk import (
    ChunkDraft,
    EmbeddedChunk,
)


log = get_logger(__name__)


class OpenAIEmbedder:
    """
    Generate embeddings for document chunks using OpenAI.
    """

    def __init__(
        self,
        batch_size: int = 64,
    ):
        settings = get_settings().embedding

        if settings.provider.value != "openai":
            raise ValueError(
                "OpenAIEmbedder requires "
                "EMBEDDING_PROVIDER=openai."
            )

        self.model_name = settings.model_name
        self.dimension = settings.dimension
        self.batch_size = batch_size

        self.client = AsyncOpenAI(
            api_key=settings.api_key,
        )

    async def embed_chunks(
        self,
        chunks: list[ChunkDraft],
    ) -> list[EmbeddedChunk]:

        if not chunks:
            return []

        embedded_chunks: list[EmbeddedChunk] = []

        for start in range(
            0,
            len(chunks),
            self.batch_size,
        ):
            batch = chunks[
                start:start + self.batch_size
            ]

            vectors = await self._embed_batch(
                batch
            )

            for chunk, vector in zip(
                batch,
                vectors,
            ):
                embedded_chunks.append(
                    EmbeddedChunk(
                        **chunk.model_dump(),
                        embedding=vector,
                    )
                )

        log.info(
            "chunks_embedded",
            chunk_count=len(embedded_chunks),
            model=self.model_name,
            dimension=self.dimension,
        )

        return embedded_chunks

    async def _embed_batch(
        self,
        chunks: list[ChunkDraft],
    ) -> list[list[float]]:

        texts = [
            chunk.text
            for chunk in chunks
        ]

        try:
            response = (
                await self.client.embeddings.create(
                    model=self.model_name,
                    input=texts,
                    dimensions=self.dimension,
                    encoding_format="float",
                )
            )

        except APIError as exc:
            raise EmbeddingError(
                "Failed to generate embeddings."
            ) from exc

        data = sorted(
            response.data,
            key=lambda item: item.index,
        )

        if len(data) != len(chunks):
            raise EmbeddingError(
                "Embedding response size does not "
                "match the number of chunks."
            )

        vectors = [
            item.embedding
            for item in data
        ]

        for vector in vectors:
            if len(vector) != self.dimension:
                raise EmbeddingError(
                    "Unexpected embedding dimension: "
                    f"expected {self.dimension}, "
                    f"received {len(vector)}."
                )

        log.debug(
            "embedding_batch_completed",
            batch_size=len(chunks),
            prompt_tokens=response.usage.prompt_tokens,
        )

        return vectors
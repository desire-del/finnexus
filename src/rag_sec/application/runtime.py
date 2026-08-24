import asyncio
from functools import cached_property, lru_cache

from langchain_core.embeddings import Embeddings

from rag_sec.config import get_settings
from rag_sec.database.manager import (
    DatabaseManager,
    get_database_manager,
)
from rag_sec.generation.generator import Generator
from rag_sec.providers import (
    get_embedding_model,
    warmup_embedding_model,
)
from rag_sec.retrieval import Retriever


class RAGRuntime:
    """Own lazily loaded RAG services and their application lifecycle."""

    def __init__(self) -> None:
        self._ready = False
        self._retrieval_ready = False
        self._warmup_lock = asyncio.Lock()

    @cached_property
    def database(self) -> DatabaseManager:
        return get_database_manager()

    @cached_property
    def embedding_model(self) -> Embeddings:
        return get_embedding_model()

    @cached_property
    def retriever(self) -> Retriever:
        return Retriever(
            embeddings=self.embedding_model,
            settings=get_settings().retrieval,
        )

    @cached_property
    def generator(self) -> Generator:
        return Generator()

    @property
    def is_ready(self) -> bool:
        return self._ready

    async def warmup(self) -> None:
        """Load and initialize every service before accepting queries."""
        await self.warmup_retrieval()

        async with self._warmup_lock:
            if self._ready:
                return

            _ = self.generator

            self._ready = True

    async def warmup_retrieval(self) -> None:
        """Initialize retrieval services without constructing the LLM."""
        async with self._warmup_lock:
            if self._retrieval_ready:
                return

            await self.database.initialize()
            await asyncio.to_thread(
                warmup_embedding_model,
                self.embedding_model,
            )
            await self.retriever.initialize()

            self._retrieval_ready = True

    async def shutdown(self) -> None:
        """Release services that were loaded during warmup."""
        if "database" not in self.__dict__:
            return

        await self.database.close()

        self._ready = False
        self._retrieval_ready = False


@lru_cache(maxsize=1)
def get_runtime() -> RAGRuntime:
    """Return the process-wide runtime without loading its services."""
    return RAGRuntime()

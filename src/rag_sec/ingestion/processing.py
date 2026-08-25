import hashlib
from dataclasses import dataclass, field
from importlib.metadata import version as package_version
from uuid import UUID

from rag_sec.config import EmbeddingSettings
from rag_sec.ingestion.chunker import SectionChunker
from rag_sec.schemas.chunk import ChunkDraft, EmbeddedChunk
from rag_sec.schemas.enums import DistanceMetric
from rag_sec.schemas.filing import FilingSection
from rag_sec.schemas.processing import ProcessingVersionCreate


@dataclass(frozen=True)
class ProcessingProfile:
    """Versioned, deterministic transformation applied to one SEC filing."""

    embedding: EmbeddingSettings
    chunker: SectionChunker
    pipeline_version: str = "ingestion-v1"
    normalization_version: str = "normalize-v1"
    chunking_strategy: str = "sec-sections+recursive-character"
    chunking_version: str = "v1"
    index_version: str = "pgvector-exact-v1"
    parser_version: str = field(default_factory=lambda: package_version("edgartools"))
    text_splitters_version: str = field(
        default_factory=lambda: package_version("langchain-text-splitters")
    )

    def fingerprint(self, content_hash: str) -> str:
        values = (
            content_hash,
            "edgartools",
            self.parser_version,
            self.normalization_version,
            self.chunking_strategy,
            self.chunking_version,
            self.text_splitters_version,
            str(self.chunker.chunk_size),
            str(self.chunker.chunk_overlap),
            self.embedding.provider.value,
            self.embedding.model_name,
            str(self.embedding.dimension),
            DistanceMetric.COSINE.value,
        )
        return hashlib.sha256("|".join(values).encode()).hexdigest()

    def version_schema(
        self,
        *,
        filing_id: UUID,
        ingestion_run_id: UUID,
        fingerprint: str,
    ) -> ProcessingVersionCreate:
        return ProcessingVersionCreate(
            filing_id=filing_id,
            ingestion_run_id=ingestion_run_id,
            pipeline_version=self.pipeline_version,
            parser_name="edgartools",
            parser_version=self.parser_version,
            normalization_version=self.normalization_version,
            chunking_strategy=self.chunking_strategy,
            chunking_version=self.chunking_version,
            embedding_provider=self.embedding.provider.value,
            embedding_model=self.embedding.model_name,
            embedding_dimension=self.embedding.dimension,
            distance_metric=DistanceMetric.COSINE,
            processing_fingerprint=fingerprint,
            index_version=self.index_version,
        )

    @staticmethod
    def normalize_sections(sections: list[FilingSection]) -> list[FilingSection]:
        normalized = []
        for section in sections:
            content = section.content.replace("\r\n", "\n").replace("\r", "\n").strip()
            if content:
                normalized.append(section.model_copy(update={"content": content}))
        return normalized

    def attach_embeddings(
        self,
        chunks: list[ChunkDraft],
        vectors: list[list[float]],
    ) -> list[EmbeddedChunk]:
        if len(chunks) != len(vectors):
            raise ValueError("Embedding count does not match chunk count.")

        embedded = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            if len(vector) != self.embedding.dimension:
                raise ValueError(
                    f"Unexpected embedding dimension: {len(vector)}. "
                    f"Expected {self.embedding.dimension}."
                )
            embedded.append(EmbeddedChunk(**chunk.model_dump(), embedding=vector))
        return embedded

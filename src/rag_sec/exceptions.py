"""
Custom exception classes for FinNexus.
"""

# BASE EXCEPTION

class FinNexusException(Exception):
    """Base exception for all FinNexus errors."""

    pass


# INGESTION

class IngestionException(FinNexusException):
    """Base exception for ingestion-related errors."""

    pass


class DocumentLoadError(IngestionException):
    """Raised when a document cannot be loaded or parsed."""

    pass


class ChunkingError(IngestionException):
    """Raised when document chunking fails."""

    pass


class EmbeddingError(IngestionException):
    """Raised when embedding generation fails."""

    pass


# STORAGE

class StorageException(FinNexusException):
    """Base exception for storage-related errors."""

    pass


class DatabaseConnectionError(StorageException):
    """Raised when the database connection fails."""

    pass


class DatabaseWriteError(StorageException):
    """Raised when writing data to the database fails."""

    pass


class DatabaseReadError(StorageException):
    """Raised when reading data from the database fails."""

    pass



# QUERY / RETRIEVAL

class RetrievalException(FinNexusException):
    """Base exception for retrieval-related errors."""

    pass


class QueryProcessingError(RetrievalException):
    """Raised when query preprocessing or analysis fails."""

    pass


class SimilaritySearchError(RetrievalException):
    """Raised when vector or similarity search fails."""

    pass


class RerankingError(RetrievalException):
    """Raised when document reranking fails."""

    pass


# GENERATION / LLM

class GenerationException(FinNexusException):
    """Base exception for generation-related errors."""

    pass


class LLMError(GenerationException):
    """Base exception for LLM-related errors."""

    pass


class LLMRateLimitError(LLMError):
    """Raised when the LLM provider rate limit is exceeded."""

    def __init__(
        self,
        message: str,
        retry_after: float | None = None,
    ):
        super().__init__(message)
        self.retry_after = retry_after


class LLMTimeoutError(LLMError):
    """Raised when an LLM request times out."""

    pass


class LLMResponseError(LLMError):
    """Raised when an LLM returns an invalid or unusable response."""

    pass
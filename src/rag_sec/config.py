from enum import Enum
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Provider enums for embedding and LLM models
class EmbeddingProvider(str, Enum):
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"


class LLMProvider(str, Enum):
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"


class AppEnvironment(str, Enum):
    DEV = "dev"
    PROD = "prod"
    STAGING = "staging"


class ObservabilityProvider(str, Enum):
    PHOENIX = "phoenix"
    NONE = "none"


# Settings class for embedding configuration
class EmbeddingSettings(BaseSettings):
    """Configuration dedicated to the embedding backend."""

    model_config = SettingsConfigDict(
        env_prefix="EMBEDDING_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    provider: EmbeddingProvider = Field(
        default=EmbeddingProvider.HUGGINGFACE,
        description="Embedding provider.",
    )
    openai_model_name: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model name.",
    )
    openai_dimension: int = Field(
        default=1536,
        gt=0,
        description="Expected OpenAI vector dimension.",
    )
    huggingface_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Hugging Face embedding model name.",
    )
    huggingface_dimension: int = Field(
        default=384,
        gt=0,
        description="Expected Hugging Face vector dimension.",
    )
    ollama_model_name: str = Field(
        default="nomic-embed-text",
        description="Ollama embedding model name.",
    )
    ollama_dimension: int = Field(
        default=768,
        gt=0,
        description="Expected Ollama vector dimension.",
    )

    api_key: str = Field(
        default="",
        description="Embedding provider API key.",
    )
    base_url: str | None = Field(
        default=None,
        description="Optional embedding API endpoint override.",
    )

    @property
    def model_name(self) -> str:
        """Return the model selected by the active provider profile."""
        match self.provider:
            case EmbeddingProvider.OPENAI:
                return self.openai_model_name
            case EmbeddingProvider.HUGGINGFACE:
                return self.huggingface_model_name
            case EmbeddingProvider.OLLAMA:
                return self.ollama_model_name

    @property
    def dimension(self) -> int:
        """Return the dimension selected by the active provider profile."""
        match self.provider:
            case EmbeddingProvider.OPENAI:
                return self.openai_dimension
            case EmbeddingProvider.HUGGINGFACE:
                return self.huggingface_dimension
            case EmbeddingProvider.OLLAMA:
                return self.ollama_dimension


# Settings class for LLM configuration
class LLMSettings(BaseSettings):
    """Configuration dedicated to the chat model backend."""

    model_config = SettingsConfigDict(
        env_prefix="LLM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    provider: LLMProvider = Field(
        default=LLMProvider.OPENAI,
        description="Chat model provider.",
    )
    model_name: str = Field(
        default="gpt-4o-mini",
        description="Chat model name.",
    )
    api_key: str = Field(
        default="",
        description="Chat model provider API key.",
    )
    base_url: str | None = Field(
        default=None,
        description="Optional chat model API endpoint override.",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Chat model sampling temperature.",
    )


class RerankerSettings(BaseSettings):
    """Configuration for local cross-encoder reranking experiments."""

    model_config = SettingsConfigDict(
        env_prefix="RERANKER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    model_name: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Sentence Transformers cross-encoder model.",
    )
    batch_size: int = Field(
        default=16,
        gt=0,
        description="Number of query-document pairs scored per batch.",
    )
    max_length: int = Field(
        default=512,
        gt=0,
        description="Maximum cross-encoder input length.",
    )


class RetrievalSettings(BaseSettings):
    """Configuration shared by production and evaluation retrieval calls."""

    model_config = SettingsConfigDict(
        env_prefix="RETRIEVAL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    mode: Literal["dense", "lexical", "hybrid", "bm25", "bm25_hybrid"] = (
        "hybrid"
    )
    top_k: int = Field(default=5, gt=0)
    dense_candidate_k: int = Field(default=20, gt=0)
    fts_candidate_k: int = Field(default=20, gt=0)
    bm25_candidate_k: int = Field(default=20, gt=0)
    hybrid_lexical_backend: Literal["fts", "bm25"] = "fts"
    rrf_k: int = Field(default=60, gt=0)
    dense_weight: float = Field(default=1.0, gt=0)
    lexical_weight: float = Field(default=1.0, gt=0)


class PhoenixSettings(BaseSettings):
    """Phoenix/OpenTelemetry exporter configuration."""

    model_config = SettingsConfigDict(
        env_prefix="PHOENIX_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    endpoint: str = Field(
        default="http://localhost:6006/v1/traces",
        description="Phoenix OTLP collector endpoint.",
    )
    project_name: str = Field(
        default="finexus",
        description="Phoenix project name.",
    )
    api_key: str = Field(
        default="",
        description="Optional Phoenix API key.",
    )
    batch: bool = Field(
        default=True,
        description="Export spans in batches.",
    )


class ObservabilitySettings(BaseSettings):
    """Application tracing configuration."""

    model_config = SettingsConfigDict(
        env_prefix="OBSERVABILITY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    provider: ObservabilityProvider = Field(
        default=ObservabilityProvider.NONE,
        description="Observability provider.",
    )
    capture_content: bool = Field(
        default=False,
        description=(
            "Capture prompt and response content in traces. "
            "Disabled by default to protect filing data."
        ),
    )
    instrument_langchain: bool = Field(
        default=False,
        description=(
            "Enable low-level LangChain spans. Disabled by "
            "default to avoid duplicate runnable spans."
        ),
    )
    instrument_openai: bool = Field(
        default=True,
        description="Enable OpenAI request, token, and cost spans.",
    )
    shutdown_timeout_seconds: float = Field(
        default=5.0,
        gt=0,
        description="Maximum time used to flush traces at shutdown.",
    )

    @property
    def config(self) -> PhoenixSettings | None:
        match self.provider:
            case ObservabilityProvider.PHOENIX:
                return PhoenixSettings()
            case ObservabilityProvider.NONE:
                return None


class EdgarSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="EDGAR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    identity: str = Field(
        min_length=3,
        description="Identity used for SEC EDGAR requests.",
    )


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: AppEnvironment = Field(default=AppEnvironment.DEV)

    log_level: str = Field(default="INFO")

    json_logging: bool = Field(default=False)

    database_url: str = Field(
        default=("postgresql+asyncpg://postgres:postgres@localhost:5433/finexus")
    )

    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)

    llm: LLMSettings = Field(default_factory=LLMSettings)

    reranker: RerankerSettings = Field(default_factory=RerankerSettings)

    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)

    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)

    # EdgarSettings reads its required identity from the EDGAR_ environment.
    edgar: EdgarSettings = Field(
        default_factory=EdgarSettings,  # type: ignore[arg-type]
    )


@lru_cache
def get_settings() -> Settings:
    """Get the application settings.

    Returns:
        Settings: The application settings.
    """
    return Settings()

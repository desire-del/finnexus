from enum import Enum
from functools import lru_cache

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
    DEV="dev"
    PROD="prod"
    STAGING="staging"

class ObservabilityProvider(str, Enum):
    PHOENIX = "phoenix"
    LANGFUSE = "langfuse"
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

class LangfuseSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LANGFUSE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra = "ignore"
    )
    host: str = Field(
        default="https://cloud.langfuse.com",
        description="The host for the Langfuse observability provider. Default is 'https://cloud.langfuse.com'."
    )
    public_key: str = Field(
        default="",
        description="The public key for the Langfuse observability provider. This is required for authentication. Please set the public key in the environment variable 'LANGFUSE_PUBLIC_KEY'."
    )
    secret_key: str = Field(
        default="",
        description="The secret key for the Langfuse observability provider. This is required for authentication. Please set the secret key in the environment variable 'LANGFUSE_SECRET_KEY'."
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
    def config(self) -> PhoenixSettings | LangfuseSettings | None:
        match self.provider:
            case ObservabilityProvider.PHOENIX:
                return PhoenixSettings()
            case ObservabilityProvider.LANGFUSE:
                return LangfuseSettings()
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

    environment: AppEnvironment = Field(
        default=AppEnvironment.DEV
    )

    log_level: str = Field(default="INFO")

    json_logging: bool = Field(default=False)

    database_url: str = Field(
        default=(
            "postgresql+asyncpg://"
            "postgres:postgres@localhost:5433/finexus"
        )
    )

    embedding: EmbeddingSettings = Field(
        default_factory=EmbeddingSettings
    )

    llm: LLMSettings = Field(
        default_factory=LLMSettings
    )

    observability: ObservabilitySettings = Field(
        default_factory=ObservabilitySettings
    )

    edgar: EdgarSettings = Field(
        default_factory=EdgarSettings
    )

@lru_cache
def get_settings() -> Settings:
    """Get the application settings.

    Returns:
        Settings: The application settings.
    """
    return Settings()

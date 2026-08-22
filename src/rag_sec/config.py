from functools import lru_cache
from enum import Enum
from typing import Union
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Provider enums for embedding and LLM models
class EmbeddingProvider(str, Enum):
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"

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
    """EmbeddingSettings is a Pydantic model that holds configuration settings for embedding models. It includes fields for the provider, model name, dimension, and API key. The class also includes validation for the API key based on the selected provider."""
    model_config = SettingsConfigDict(
        env_prefix="EMBEDDING_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra = "ignore"
    )

    provider: EmbeddingProvider = Field(
        default=EmbeddingProvider.OPENAI,
        description="The provider for the embedding model. Options are 'openai' or 'huggingface'."
    )
    model_name: str = Field(
        default="text-embedding-3-small",
        description="The name of the embedding model to use. For OpenAI, options include 'text-embedding-3-small', 'text-embedding-3-large', etc. For HuggingFace, options include 'sentence-transformers/all-MiniLM-L6-v2', 'sentence-transformers/all-mpnet-base-v2', etc."
    )
    dimension: int = Field(
        default=1536,
        description="The dimension of the embedding vector. For OpenAI's 'text-embedding-3-small', the dimension is 1536. For HuggingFace's 'sentence-transformers/all-MiniLM-L6-v2', the dimension is 384."
    )

    api_key: str = Field(
        default="",
        description="The API key for the embedding provider. This is required for OpenAI and HuggingFace. For OpenAI, you can set the API key in the environment variable 'OPENAI_API_KEY'. For HuggingFace, you can set the API key in the environment variable 'HUGGINGFACE_API_KEY'."
    )
    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, value, info):
        provider = info.data.get("provider")
        if provider in [EmbeddingProvider.OPENAI] and not value:
            raise ValueError(f"API key is required for provider '{provider.value}'. Please set the API key in the environment variable 'OPENAI_API_KEY'.")
        return value

# Settings class for LLM configuration
class LLMSettings(BaseSettings):
    """LLMSettings is a Pydantic model that holds configuration settings for LLM models. It includes fields for the provider, model name, dimension, and API key. The class also includes validation for the API key based on the selected provider."""
    model_config = SettingsConfigDict(
        env_prefix="LLM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra = "ignore"
    )

    provider: LLMProvider = Field(
        default=LLMProvider.OPENAI,
        description="The provider for the LLM model. Options are 'openai', 'huggingface', or 'ollama'."
    )
    model_name: str = Field(
        default="gpt-4o",
        description="The name of the LLM model to use. For OpenAI, options include 'gpt-4o', 'gpt-4o-mini', etc. For HuggingFace, options include 'bigscience/bloom', 'facebook/opt-1.3b', etc. For Ollama, options include 'ollama/llama2-7b', 'ollama/llama2-13b', etc."
    )
    api_key: str = Field(
        default="",
        description="The API key for the LLM provider. This is required for OpenAI and HuggingFace. For OpenAI, you can set the API key in the environment variable 'OPENAI_API_KEY'. For HuggingFace, you can set the API key in the environment variable 'HUGGINGFACE_API_KEY'."
    )
    base_url: str = Field(
        default="https://localhost:11434",
        description="The base URL for the LLM provider's API. This is optional and can be used to override the default API endpoint. Default is 'https://localhost:11434' for Ollama. For OpenAI and HuggingFace, the default endpoints are used."
    )

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, value, info):
        provider = info.data.get("provider")
        if provider in [LLMProvider.OPENAI] and not value:
            raise ValueError(f"API key is required for provider '{provider.value}'. Please set the API key in the environment variable 'OPENAI_API_KEY'.")
        return value

class PhoenixSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PHOENIX_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra = "ignore"
    )
    endpoint: str = Field(
        default="http://localhost:6060",
        description="The endpoint for the Phoenix observability provider. Default is 'http://localhost:6060'."
    )
    project_name: str = Field(
        default="finexus",
        description="The project name for the Phoenix observability provider. Default is 'finexus'."
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
    model_config = SettingsConfigDict(
        env_prefix="OBSERVABILITY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra = "ignore"
    )
    provider: ObservabilityProvider = Field(
        default=ObservabilityProvider.NONE,
        description="The provider for observability. Options are 'phoenix', 'langfuse', or 'none'."
    )


    @property
    def config(self)->Union[PhoenixSettings, LangfuseSettings, None]:
        match self.provider:
            case ObservabilityProvider.PHOENIX:
                return PhoenixSettings()
            case ObservabilityProvider.LANGFUSE:
                return LangfuseSettings()
            case ObservabilityProvider.NONE:
                return None
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra = "ignore"
    )
    environment: AppEnvironment = Field(
        default=AppEnvironment.DEV,
        description="The application environment. Options are 'dev', 'prod', or 'staging'."
    )
    # App settings
    log_level: str = Field(
        default="INFO",
        description="The log level for the application. Options are 'DEBUG', 'INFO', 'WARNING', 'ERROR', or 'CRITICAL'. Default is 'INFO'."
    )

    json_logging: bool = Field(
        default=False,
        description="Whether to enable JSON logging. Default is False. Set to True in production."
    )

    # Database settings
    database_url: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/finexus",
        description="The database URL for the application. Default is 'postgresql://postgres:postgres@localhost:5432/finexus'."
    )

    # nested settings for embedding, LLM, and observability
    embedding: EmbeddingSettings = Field(
        default_factory=EmbeddingSettings,
        description="The settings for the embedding model."
    )
    llm: LLMSettings = Field(
        default_factory=LLMSettings,
        description="The settings for the LLM model."
    )
    observability: ObservabilitySettings = Field(
        default_factory=ObservabilitySettings,
        description="The settings for observability."
    )


@lru_cache()
def get_settings() -> Settings:
    """Get the application settings.

    Returns:
        Settings: The application settings.
    """
    return Settings()
        
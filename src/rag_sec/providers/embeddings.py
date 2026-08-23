from functools import lru_cache

import tiktoken
from langchain.embeddings import init_embeddings
from langchain_core.embeddings import Embeddings

from rag_sec.config import get_settings


@lru_cache(maxsize=1)
def get_embedding_model() -> Embeddings:
    """Build the configured embedding provider on first use."""
    settings = get_settings().embedding
    kwargs = {}

    if settings.provider.value == "openai":
        if settings.api_key:
            kwargs["api_key"] = settings.api_key

        if settings.base_url:
            kwargs["base_url"] = settings.base_url

        kwargs["dimensions"] = settings.dimension
    elif settings.provider.value == "ollama" and settings.base_url:
        kwargs["base_url"] = settings.base_url

    return init_embeddings(
        model=settings.model_name,
        provider=settings.provider.value,
        **kwargs,
    )


def warmup_embedding_model(model: Embeddings) -> None:
    """Preload provider-local resources without calling a remote API."""
    settings = get_settings().embedding

    if settings.provider.value == "huggingface":
        model.embed_query("warmup")
        return

    if settings.provider.value != "openai":
        return

    if not getattr(model, "check_embedding_ctx_length", False):
        return

    if not getattr(model, "tiktoken_enabled", False):
        return

    model_name = getattr(model, "tiktoken_model_name", None) or settings.model_name

    try:
        tiktoken.encoding_for_model(model_name)
    except KeyError:
        tiktoken.get_encoding("cl100k_base")

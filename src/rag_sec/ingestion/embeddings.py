from functools import lru_cache

from langchain.embeddings import init_embeddings
from langchain_core.embeddings import Embeddings

from rag_sec.config import get_settings


@lru_cache(maxsize=1)
def get_embedding_model() -> Embeddings:
    settings = get_settings().embedding

    kwargs = {}

    if settings.api_key:
        kwargs["api_key"] = settings.api_key

    if settings.base_url:
        kwargs["base_url"] = settings.base_url

    # OpenAI supports configurable output dimensions
    if settings.provider.value == "openai":
        kwargs["dimensions"] = settings.dimension

    return init_embeddings(
        model=settings.model_name,
        provider=settings.provider.value,
        **kwargs,
    )

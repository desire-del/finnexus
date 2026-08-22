from functools import lru_cache

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel

from rag_sec.config import get_settings


@lru_cache(maxsize=1)
def get_chat_model() -> BaseChatModel:
    settings = get_settings().llm

    kwargs = {
        "temperature": settings.temperature,
    }

    if settings.api_key:
        kwargs["api_key"] = settings.api_key

    if settings.base_url:
        kwargs["base_url"] = settings.base_url

    return init_chat_model(
        model=settings.model_name,
        model_provider=settings.provider.value,
        **kwargs,
    )

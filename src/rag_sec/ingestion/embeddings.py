# src/rag_sec/ingestion/embeddings.py

from functools import lru_cache

from langchain_openai import OpenAIEmbeddings

from rag_sec.config import get_settings


@lru_cache(maxsize=1)
def get_embedding_model() -> OpenAIEmbeddings:

    settings = get_settings().embedding

    return OpenAIEmbeddings(
        model=settings.model_name,
        dimensions=settings.dimension,
        api_key=settings.api_key,
        chunk_size=64,
    )
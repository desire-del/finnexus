import asyncio
from types import SimpleNamespace

import pytest

from rag_sec import observability
from rag_sec.application import query as query_module


class StubEmbeddingModel:
    def __init__(self, vector: list[float]) -> None:
        self.vector = vector
        self.queries = []

    async def aembed_query(self, query: str) -> list[float]:
        self.queries.append(query)
        return self.vector


def embedding_settings(dimension: int):
    return SimpleNamespace(
        embedding=SimpleNamespace(
            provider=SimpleNamespace(value="test"),
            model_name="test-embedding",
            dimension=dimension,
        ),
        observability=SimpleNamespace(capture_content=False),
    )


def test_embed_query_returns_validated_vector(monkeypatch):
    model = StubEmbeddingModel([0.1, 0.2, 0.3])

    monkeypatch.setattr(
        observability,
        "configure_observability",
        lambda: None,
    )
    monkeypatch.setattr(
        query_module,
        "get_settings",
        lambda: embedding_settings(3),
    )

    vector = asyncio.run(query_module.embed_query("  test query  ", model=model))

    assert vector == [0.1, 0.2, 0.3]
    assert model.queries == ["test query"]


def test_embed_query_rejects_unexpected_dimension(monkeypatch):
    model = StubEmbeddingModel([0.1, 0.2])

    monkeypatch.setattr(
        observability,
        "configure_observability",
        lambda: None,
    )
    monkeypatch.setattr(
        query_module,
        "get_settings",
        lambda: embedding_settings(3),
    )

    with pytest.raises(ValueError, match="expected 3, received 2"):
        asyncio.run(query_module.embed_query("test query", model=model))

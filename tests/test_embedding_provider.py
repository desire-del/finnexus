from types import SimpleNamespace

from rag_sec.providers import embeddings as embedding_provider


def settings(provider: str, model_name: str = "embedding-model"):
    return SimpleNamespace(
        embedding=SimpleNamespace(
            provider=SimpleNamespace(value=provider),
            model_name=model_name,
        )
    )


def test_warmup_preloads_openai_tokenizer(monkeypatch):
    loaded_models = []
    model = SimpleNamespace(
        check_embedding_ctx_length=True,
        tiktoken_enabled=True,
        tiktoken_model_name="tokenizer-model",
    )

    monkeypatch.setattr(
        embedding_provider,
        "get_settings",
        lambda: settings("openai"),
    )
    monkeypatch.setattr(
        embedding_provider.tiktoken,
        "encoding_for_model",
        loaded_models.append,
    )

    embedding_provider.warmup_embedding_model(model)

    assert loaded_models == ["tokenizer-model"]


def test_warmup_ignores_non_openai_provider(monkeypatch):
    model = SimpleNamespace(
        check_embedding_ctx_length=True,
        tiktoken_enabled=True,
    )

    monkeypatch.setattr(
        embedding_provider,
        "get_settings",
        lambda: settings("ollama"),
    )
    monkeypatch.setattr(
        embedding_provider.tiktoken,
        "encoding_for_model",
        lambda _model: (_ for _ in ()).throw(
            AssertionError("Tokenizer must not be loaded.")
        ),
    )

    embedding_provider.warmup_embedding_model(model)

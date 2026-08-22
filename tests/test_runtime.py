import asyncio

from rag_sec.application import runtime as runtime_module


class StubDatabase:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def initialize(self) -> None:
        self.events.append("database.initialize")

    async def close(self) -> None:
        self.events.append("database.close")


class StubRetriever:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def initialize(self) -> None:
        self.events.append("retrieval.warmup")


class StubGenerator:
    pass


class StubEmbeddings:
    pass


def test_runtime_services_are_lazy_and_cached(monkeypatch):
    events = []
    database = StubDatabase(events)
    embeddings = StubEmbeddings()
    retriever = StubRetriever(events)
    generator = StubGenerator()

    monkeypatch.setattr(
        runtime_module,
        "get_database_manager",
        lambda: events.append("database.load") or database,
    )
    monkeypatch.setattr(
        runtime_module,
        "get_embedding_model",
        lambda: events.append("embeddings.load") or embeddings,
    )
    monkeypatch.setattr(
        runtime_module,
        "warmup_embedding_model",
        lambda _model: events.append("embeddings.warmup"),
    )
    monkeypatch.setattr(
        runtime_module,
        "Retriever",
        lambda **_options: events.append("retriever.load") or retriever,
    )
    monkeypatch.setattr(
        runtime_module,
        "Generator",
        lambda: events.append("generator.load") or generator,
    )

    runtime = runtime_module.RAGRuntime()

    assert not hasattr(runtime.warmup, "__wrapped__")
    assert not hasattr(runtime.shutdown, "__wrapped__")
    assert events == []
    assert runtime.database is database
    assert runtime.database is database
    assert runtime.embedding_model is embeddings
    assert runtime.embedding_model is embeddings
    assert runtime.retriever is retriever
    assert runtime.retriever is retriever
    assert runtime.generator is generator
    assert runtime.generator is generator
    assert events == [
        "database.load",
        "embeddings.load",
        "retriever.load",
        "generator.load",
    ]


def test_warmup_prepares_services_once_and_shutdown_closes_database(
    monkeypatch,
):
    events = []
    database = StubDatabase(events)
    embeddings = StubEmbeddings()
    retriever = StubRetriever(events)

    monkeypatch.setattr(
        runtime_module,
        "get_database_manager",
        lambda: events.append("database.load") or database,
    )
    monkeypatch.setattr(
        runtime_module,
        "get_embedding_model",
        lambda: events.append("embeddings.load") or embeddings,
    )
    monkeypatch.setattr(
        runtime_module,
        "warmup_embedding_model",
        lambda _model: events.append("embeddings.warmup"),
    )
    monkeypatch.setattr(
        runtime_module,
        "Retriever",
        lambda **_options: events.append("retriever.load") or retriever,
    )
    monkeypatch.setattr(
        runtime_module,
        "Generator",
        lambda: events.append("generator.load") or StubGenerator(),
    )

    runtime = runtime_module.RAGRuntime()

    async def exercise_runtime() -> None:
        await runtime.warmup()
        await runtime.warmup()
        await runtime.shutdown()

    asyncio.run(exercise_runtime())

    assert events == [
        "database.load",
        "database.initialize",
        "embeddings.load",
        "embeddings.warmup",
        "retriever.load",
        "retrieval.warmup",
        "generator.load",
        "database.close",
    ]
    assert runtime.is_ready is False

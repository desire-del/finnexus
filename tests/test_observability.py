import asyncio
from types import SimpleNamespace

from openinference.semconv.trace import (
    OpenInferenceSpanKindValues,
    SpanAttributes,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode

from rag_sec import observability
from rag_sec.config import (
    ObservabilityProvider,
    PhoenixSettings,
)


class StubInstrumentor:
    def __init__(self):
        self.options = None

    def instrument(self, **options):
        self.options = options


def test_track_preserves_sync_result_when_disabled(
    monkeypatch,
):
    monkeypatch.setattr(
        observability,
        "configure_observability",
        lambda: None,
    )

    @observability.track(
        name="test.sync",
        phase=observability.Phase.QUERY,
    )
    def operation(value: int) -> int:
        return value * 2

    with observability.trace_source("test"):
        assert operation(21) == 42


def test_track_preserves_async_result_when_disabled(
    monkeypatch,
):
    monkeypatch.setattr(
        observability,
        "configure_observability",
        lambda: None,
    )

    @observability.track(
        name="test.async",
        phase=observability.Phase.GENERATION,
    )
    async def operation(value: int) -> int:
        return value + 1

    with observability.trace_source("test"):
        assert asyncio.run(operation(41)) == 42


def test_set_attributes_without_active_span():
    observability.set_span_attributes(
        {
            "rag.test.value": 42,
            "rag.test.empty": None,
        }
    )


def test_configure_phoenix_with_private_content(
    monkeypatch,
):
    tracer_provider = object()
    register_options = {}
    langchain_instrumentor = StubInstrumentor()
    openai_instrumentor = StubInstrumentor()

    settings = SimpleNamespace(
        observability=SimpleNamespace(
            provider=ObservabilityProvider.PHOENIX,
            config=PhoenixSettings(
                endpoint="http://phoenix.test",
                project_name="test-project",
                api_key="secret",
                batch=False,
                _env_file=None,
            ),
            capture_content=False,
            instrument_langchain=True,
            instrument_openai=True,
            shutdown_timeout_seconds=1.0,
        )
    )

    def fake_register(**options):
        register_options.update(options)
        return tracer_provider

    monkeypatch.setattr(
        observability,
        "get_settings",
        lambda: settings,
    )
    monkeypatch.setattr(
        observability,
        "register",
        fake_register,
    )
    monkeypatch.setattr(
        observability,
        "LangChainInstrumentor",
        lambda: langchain_instrumentor,
    )
    monkeypatch.setattr(
        observability,
        "OpenAIInstrumentor",
        lambda: openai_instrumentor,
    )

    observability.configure_observability.cache_clear()

    try:
        configured = observability.configure_observability()
    finally:
        observability.configure_observability.cache_clear()

    assert configured is tracer_provider
    assert register_options["project_name"] == "test-project"
    assert register_options["batch"] is False

    langchain_config = langchain_instrumentor.options["config"]
    openai_config = openai_instrumentor.options["config"]

    assert langchain_config.hide_inputs is True
    assert langchain_config.hide_outputs is True
    assert openai_config.hide_embeddings_vectors is True


def test_track_sets_openinference_metadata_and_status(
    monkeypatch,
):
    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

    monkeypatch.setattr(
        observability,
        "configure_observability",
        lambda: tracer_provider,
    )
    monkeypatch.setattr(
        observability.trace,
        "get_tracer",
        tracer_provider.get_tracer,
    )

    @observability.track(
        name="test.retrieval",
        phase=observability.Phase.RETRIEVAL,
        span_kind=(OpenInferenceSpanKindValues.RETRIEVER),
    )
    def operation() -> int:
        observability.set_span_input({"query_length": 12})
        observability.set_span_output({"document_count": 3})
        return 3

    assert operation() == 3

    spans = exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]

    assert span.status.status_code is StatusCode.OK
    assert (
        span.attributes[SpanAttributes.OPENINFERENCE_SPAN_KIND]
        == OpenInferenceSpanKindValues.RETRIEVER.value
    )
    assert span.attributes[SpanAttributes.INPUT_MIME_TYPE] == "application/json"
    assert span.attributes[SpanAttributes.OUTPUT_VALUE] == '{"document_count": 3}'

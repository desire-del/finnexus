import inspect
import json
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from enum import Enum
from functools import lru_cache, wraps
from typing import Any

from openinference.instrumentation import TraceConfig
from openinference.instrumentation.langchain import (
    LangChainInstrumentor,
)
from openinference.instrumentation.openai import (
    OpenAIInstrumentor,
)
from openinference.semconv.trace import (
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import StatusCode
from phoenix.otel import register, using_tags

from rag_sec.config import (
    ObservabilityProvider,
    PhoenixSettings,
    get_settings,
)
from rag_sec.logging import get_logger

log = get_logger(__name__)

_trace_source: ContextVar[str] = ContextVar(
    "trace_source",
    default="unknown",
)


class Phase(str, Enum):
    INGESTION = "ingestion"
    RETRIEVAL = "retrieval"
    QUERY = "query"
    GENERATION = "generation"


@lru_cache(maxsize=1)
def configure_observability() -> TracerProvider | None:
    settings = get_settings().observability

    if settings.provider is ObservabilityProvider.NONE:
        log.info("observability_disabled")
        return None

    if settings.provider is not ObservabilityProvider.PHOENIX:
        raise NotImplementedError(
            "Only the 'phoenix' and 'none' observability "
            "providers are currently supported."
        )

    phoenix_settings = settings.config

    if not isinstance(phoenix_settings, PhoenixSettings):
        raise TypeError("Phoenix observability settings are unavailable.")

    tracer_provider = register(
        project_name=phoenix_settings.project_name,
        endpoint=phoenix_settings.endpoint,
        protocol="http/protobuf",
        api_key=phoenix_settings.api_key or None,
        batch=phoenix_settings.batch,
        auto_instrument=False,
        verbose=False,
    )

    trace_config = TraceConfig(
        hide_inputs=not settings.capture_content,
        hide_outputs=not settings.capture_content,
        hide_embeddings_vectors=True,
    )

    if settings.instrument_langchain:
        LangChainInstrumentor().instrument(
            tracer_provider=tracer_provider,
            config=trace_config,
        )

    if settings.instrument_openai:
        OpenAIInstrumentor().instrument(
            tracer_provider=tracer_provider,
            config=trace_config,
        )

    log.info(
        "observability_configured",
        provider=settings.provider.value,
        project_name=phoenix_settings.project_name,
        endpoint=phoenix_settings.endpoint,
        capture_content=settings.capture_content,
        instrument_langchain=settings.instrument_langchain,
        instrument_openai=settings.instrument_openai,
    )

    return tracer_provider


def shutdown_observability() -> None:
    tracer_provider = configure_observability()

    if tracer_provider is None:
        return

    settings = get_settings().observability
    timeout_millis = int(settings.shutdown_timeout_seconds * 1000)

    flushed = tracer_provider.force_flush(timeout_millis=timeout_millis)

    if not flushed:
        log.warning(
            "observability_flush_timed_out",
            timeout_millis=timeout_millis,
        )

    tracer_provider.shutdown()
    log.info("observability_shutdown")


@contextmanager
def trace_source(source: str) -> Iterator[None]:
    token = _trace_source.set(source)

    try:
        yield
    finally:
        _trace_source.reset(token)


def set_span_attributes(
    attributes: Mapping[str, Any],
) -> None:
    span = trace.get_current_span()

    if not span.is_recording():
        return

    for key, value in attributes.items():
        if value is not None:
            span.set_attribute(key, value)


def set_span_input(value: Mapping[str, Any]) -> None:
    _set_span_io(
        value_attribute=SpanAttributes.INPUT_VALUE,
        mime_attribute=SpanAttributes.INPUT_MIME_TYPE,
        value=value,
    )


def set_span_output(value: Mapping[str, Any]) -> None:
    _set_span_io(
        value_attribute=SpanAttributes.OUTPUT_VALUE,
        mime_attribute=SpanAttributes.OUTPUT_MIME_TYPE,
        value=value,
    )


def track(
    name: str | None = None,
    phase: Phase | None = None,
    tags: list[str] | None = None,
    span_kind: OpenInferenceSpanKindValues = (OpenInferenceSpanKindValues.CHAIN),
) -> Callable:
    def decorator(function: Callable) -> Callable:
        def span_data() -> tuple[str, list[str], str]:
            span_name = name or function.__name__
            final_tags = list(tags or [])

            if phase is not None:
                final_tags.append(f"phase:{phase.value}")

            source = _trace_source.get()

            if source != "unknown":
                final_tags.append(f"source:{source}")

            return span_name, final_tags, source

        @wraps(function)
        async def async_wrapper(*args, **kwargs):
            configure_observability()
            tracer = trace.get_tracer(function.__module__)
            span_name, final_tags, source = span_data()

            with (
                using_tags(final_tags),
                tracer.start_as_current_span(span_name) as span,
            ):
                span.set_attribute(
                    SpanAttributes.OPENINFERENCE_SPAN_KIND,
                    span_kind.value,
                )
                _set_common_attributes(phase, source)
                result = await function(*args, **kwargs)
                span.set_status(StatusCode.OK)
                return result

        @wraps(function)
        def sync_wrapper(*args, **kwargs):
            configure_observability()
            tracer = trace.get_tracer(function.__module__)
            span_name, final_tags, source = span_data()

            with (
                using_tags(final_tags),
                tracer.start_as_current_span(span_name) as span,
            ):
                span.set_attribute(
                    SpanAttributes.OPENINFERENCE_SPAN_KIND,
                    span_kind.value,
                )
                _set_common_attributes(phase, source)
                result = function(*args, **kwargs)
                span.set_status(StatusCode.OK)
                return result

        if inspect.iscoroutinefunction(function):
            return async_wrapper

        return sync_wrapper

    return decorator


def _set_common_attributes(
    phase: Phase | None,
    source: str,
) -> None:
    attributes: dict[str, str] = {}

    if phase is not None:
        attributes["rag.phase"] = phase.value

    if source != "unknown":
        attributes["rag.source"] = source

    set_span_attributes(attributes)


def _set_span_io(
    *,
    value_attribute: str,
    mime_attribute: str,
    value: Mapping[str, Any],
) -> None:
    set_span_attributes(
        {
            value_attribute: json.dumps(
                value,
                default=str,
                sort_keys=True,
            ),
            mime_attribute: OpenInferenceMimeTypeValues.JSON.value,
        }
    )

# src/observability.py

from contextlib import contextmanager
from contextvars import ContextVar
from enum import Enum
from functools import lru_cache, wraps
import inspect
from typing import Optional, List

from opentelemetry import trace
from phoenix.otel import register, using_tags

from rag_sec.config import get_settings
from rag_sec.logging import get_logger


log = get_logger(__name__)


# CONTEXT

# Stores the source of the current execution flow.
# - "ingestion_script"
# - "rest_api"
# - "evaluation"
# - "notebook"
_trace_source: ContextVar[str] = ContextVar(
    "trace_source",
    default="unknown",
)



# PHASES
class Phase(str, Enum):
    INGESTION = "ingestion"
    RETRIEVAL = "retrieval"
    QUERY = "query"
    GENERATION = "generation"



# PHOENIX CONFIGURATION
@lru_cache(maxsize=1)
def configure_observability():
    """
    Configure Phoenix tracing once for the current Python process.
    """

    settings = get_settings()
    phoenix_settings = settings.observability.config

    if phoenix_settings is None:
        log.info("observability_disabled")
        return None

    tracer_provider = register(
        project_name=phoenix_settings.project_name,
        endpoint=phoenix_settings.endpoint,
        protocol="http/protobuf",
        auto_instrument=True,
    )

    log.info(
        "observability_configured",
        provider="phoenix",
        project_name=phoenix_settings.project_name,
        endpoint=phoenix_settings.endpoint,
    )

    return tracer_provider


# TRACE SOURCE
@contextmanager
def trace_source(source: str):
    """
    Temporarily define the source of the current execution flow.

    Example:
        with trace_source("ingestion_script"):
            ingest_document(...)
    """

    token = _trace_source.set(source)

    try:
        yield
    finally:
        _trace_source.reset(token)



# TRACKING DECORATOR
def track(
    name: Optional[str] = None,
    phase: Optional[Phase] = None,
    tags: Optional[List[str]] = None,
):
    """
    Create a Phoenix/OpenTelemetry span around a function.

    Works with both synchronous and asynchronous functions.
    """

    def decorator(func):

        def get_span_data():
            span_name = name or func.__name__

            final_tags = list(tags or [])

            if phase:
                final_tags.append(f"phase:{phase.value}")

            source = _trace_source.get()

            if source != "unknown":
                final_tags.append(f"source:{source}")

            return span_name, final_tags, source

        @wraps(func)
        async def async_wrapper(*args, **kwargs):

            configure_observability()

            tracer = trace.get_tracer(func.__module__)

            span_name, final_tags, source = get_span_data()

            with using_tags(final_tags):
                with tracer.start_as_current_span(span_name) as span:

                    if phase:
                        span.set_attribute(
                            "finexus.phase",
                            phase.value,
                        )

                    if source != "unknown":
                        span.set_attribute(
                            "finexus.source",
                            source,
                        )

                    return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):

            configure_observability()

            tracer = trace.get_tracer(func.__module__)

            span_name, final_tags, source = get_span_data()

            with using_tags(final_tags):
                with tracer.start_as_current_span(span_name) as span:

                    if phase:
                        span.set_attribute(
                            "finexus.phase",
                            phase.value,
                        )

                    if source != "unknown":
                        span.set_attribute(
                            "finexus.source",
                            source,
                        )

                    return func(*args, **kwargs)

        if inspect.iscoroutinefunction(func):
            return async_wrapper

        return sync_wrapper

    return decorator
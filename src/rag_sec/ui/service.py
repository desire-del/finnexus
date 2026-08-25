from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from concurrent.futures import Future
from threading import Lock, Thread
from typing import Any, TypeVar

from rag_sec.application import (
    AvailableFiling,
    RAGRuntime,
    answer_query,
    get_runtime,
    list_available_filings,
)
from rag_sec.generation.models import RAGAnswer
from rag_sec.ingestion.pipeline import IngestionPipeline
from rag_sec.observability import (
    configure_observability,
    shutdown_observability,
    trace_source,
)
from rag_sec.schemas.ingestion import IngestionResult

Result = TypeVar("Result")


class RAGService:
    """Expose the asynchronous RAG runtime to synchronous UI frameworks."""

    def __init__(self, runtime: RAGRuntime | None = None) -> None:
        configure_observability()

        self.runtime = runtime or get_runtime()
        self._loop = asyncio.new_event_loop()
        self._thread = Thread(
            target=self._serve,
            name="rag-sec-async-runtime",
            daemon=True,
        )
        self._close_lock = Lock()
        self._ingestion_lock = asyncio.Lock()
        self._ingestion_pipeline: IngestionPipeline | None = None
        self._closed = False
        self._ready = False
        self._thread.start()

    @property
    def is_ready(self) -> bool:
        return self._ready and not self._closed

    def _serve(self) -> None:
        asyncio.set_event_loop(self._loop)

        try:
            self._loop.run_forever()
        finally:
            pending = asyncio.all_tasks(self._loop)

            for task in pending:
                task.cancel()

            if pending:
                self._loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )

            self._loop.close()

    def _submit(
        self,
        coroutine: Coroutine[Any, Any, Result],
    ) -> Result:
        if self._closed:
            coroutine.close()
            raise RuntimeError("The RAG service is closed.")

        future: Future[Result] = asyncio.run_coroutine_threadsafe(
            coroutine,
            self._loop,
        )
        return future.result()

    def warmup(self) -> None:
        """Prepare the database, embedding model, retriever, and LLM."""
        self._submit(self.runtime.warmup())
        self._ready = True

    async def _answer(
        self,
        question: str,
        *,
        ticker: str,
        form_type: str,
    ) -> RAGAnswer:
        with trace_source("streamlit"):
            return await answer_query(
                self.runtime,
                question,
                ticker=ticker,
                form_type=form_type,
            )

    def answer(
        self,
        question: str,
        *,
        ticker: str,
        form_type: str,
    ) -> RAGAnswer:
        if not self.is_ready:
            self.warmup()

        return self._submit(
            self._answer(
                question,
                ticker=ticker,
                form_type=form_type,
            )
        )

    async def _ingest(
        self,
        identifier: str,
        *,
        form_type: str,
    ) -> IngestionResult:
        async with self._ingestion_lock:
            if self._ingestion_pipeline is None:
                self._ingestion_pipeline = IngestionPipeline()

            normalized = identifier.strip().upper()
            normalized_identifier: str | int = (
                int(normalized) if normalized.isdigit() else normalized
            )

            with trace_source("streamlit.ingestion"):
                return await self._ingestion_pipeline.ingest_latest(
                    normalized_identifier,
                    form_type=form_type.strip().upper(),
                )

    def ingest(
        self,
        identifier: str,
        *,
        form_type: str,
    ) -> IngestionResult:
        """Ingest the latest filing for a ticker or CIK."""
        if not identifier.strip():
            raise ValueError("A ticker or CIK is required.")

        if not self.is_ready:
            self.warmup()

        return self._submit(
            self._ingest(
                identifier,
                form_type=form_type,
            )
        )

    def list_filings(self, *, limit: int = 200) -> list[AvailableFiling]:
        """Return filings available to the active embedding profile."""
        if not self.is_ready:
            self.warmup()

        return self._submit(
            list_available_filings(
                self.runtime,
                limit=limit,
            )
        )

    def close(self) -> None:
        """Release the application runtime and its background event loop."""
        with self._close_lock:
            if self._closed:
                return

            try:
                self._submit(self.runtime.shutdown())
            finally:
                self._ready = False
                self._closed = True
                self._loop.call_soon_threadsafe(self._loop.stop)
                self._thread.join(timeout=5)
                shutdown_observability()

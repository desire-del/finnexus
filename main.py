import asyncio

from rag_sec.application import answer_query, get_runtime
from rag_sec.generation.generator import RAGAnswer
from rag_sec.observability import (
    configure_observability,
    shutdown_observability,
    trace_source,
)


def render_answer(result: RAGAnswer) -> None:
    print("\nANSWER\n")
    print(result.answer)
    print("\nSOURCES\n")

    for source in result.sources:
        print(
            f"{source.source_id} | "
            f"{source.company_name} | "
            f"{source.form_type} | "
            f"{source.item} | "
            f"{source.accession_number}"
        )
        print(source.source_url)
        print()


async def main() -> None:
    runtime = get_runtime()

    try:
        await runtime.warmup()
        result = await answer_query(
            runtime,
            "How did foreign exchange affect Apple's results?",
            ticker="AAPL",
            form_type="10-K",
        )
        render_answer(result)
    finally:
        await runtime.shutdown()


if __name__ == "__main__":
    configure_observability()

    try:
        with trace_source("cli"):
            asyncio.run(main())
    finally:
        shutdown_observability()

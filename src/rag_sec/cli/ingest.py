from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence

from rag_sec.database.manager import get_database_manager
from rag_sec.ingestion.pipeline import IngestionPipeline
from rag_sec.observability import (
    configure_observability,
    shutdown_observability,
    trace_source,
)
from rag_sec.providers import warmup_embedding_model
from rag_sec.schemas.ingestion import IngestionResult


def positive_integer(value: str) -> int:
    """Parse a strictly positive command-line integer."""
    parsed = int(value)

    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")

    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-sec-ingest",
        description="Ingest the latest SEC filing for a company.",
    )
    parser.add_argument(
        "identifier",
        help="Company ticker or SEC CIK (for example AAPL or 320193).",
    )
    parser.add_argument(
        "--form-type",
        default="10-K",
        help="SEC filing form to ingest (default: 10-K).",
    )
    parser.add_argument(
        "--chunk-size",
        type=positive_integer,
        default=800,
        help="Maximum chunk size in characters (default: 800).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=100,
        help="Overlap between chunks in characters (default: 100).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Print the ingestion result as JSON.",
    )
    return parser


def validate_arguments(
    parser: argparse.ArgumentParser,
    arguments: argparse.Namespace,
) -> None:
    if arguments.chunk_overlap < 0:
        parser.error("--chunk-overlap must be zero or greater")

    if arguments.chunk_overlap >= arguments.chunk_size:
        parser.error("--chunk-overlap must be smaller than --chunk-size")


def normalize_identifier(identifier: str) -> str | int:
    """Keep tickers as strings and pass numeric CIK values as integers."""
    normalized = identifier.strip()
    return int(normalized) if normalized.isdigit() else normalized.upper()


async def run_ingestion(arguments: argparse.Namespace) -> IngestionResult:
    """Run one ingestion while owning the database lifecycle."""
    database = get_database_manager()

    try:
        await database.initialize()

        pipeline = IngestionPipeline(
            chunk_size=arguments.chunk_size,
            chunk_overlap=arguments.chunk_overlap,
        )
        await asyncio.to_thread(
            warmup_embedding_model,
            pipeline.embeddings,
        )

        return await pipeline.ingest_latest(
            normalize_identifier(arguments.identifier),
            form_type=arguments.form_type.strip().upper(),
        )
    finally:
        await database.close()


def render_result(result: IngestionResult, *, as_json: bool) -> None:
    if as_json:
        print(
            json.dumps(
                result.model_dump(mode="json"),
                indent=2,
            )
        )
        return

    print("\nINGESTION RESULT\n")
    print(f"Run ID: {result.ingestion_run_id}")
    print(f"Status: {result.status.value}")
    print(f"Discovered: {result.filings_discovered}")
    print(f"Processed: {result.filings_processed}")
    print(f"Skipped: {result.filings_skipped}")
    print(f"Failed: {result.filings_failed}")

    if result.processed_filing_ids:
        print("Processed filing IDs:")
        for filing_id in result.processed_filing_ids:
            print(f"  - {filing_id}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    validate_arguments(parser, arguments)
    observability_ready = False

    try:
        configure_observability()
        observability_ready = True

        with trace_source("cli.ingestion"):
            result = asyncio.run(run_ingestion(arguments))

        render_result(result, as_json=arguments.json_output)
        return 1 if result.filings_failed else 0
    except KeyboardInterrupt:
        print("Ingestion interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI boundary reports all failures.
        print(f"Ingestion failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if observability_ready:
            shutdown_observability()


if __name__ == "__main__":
    raise SystemExit(main())

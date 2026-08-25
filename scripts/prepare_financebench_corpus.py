"""Resolve and ingest the SEC corpus required by FinanceBench."""

import asyncio

from rag_sec.database.manager import get_database_manager
from rag_sec.evaluation.corpus.financebench import (
    prepare_financebench_corpus,
)
from rag_sec.evaluation.suite import FinanceBenchSuite
from rag_sec.ingestion.pipeline import IngestionPipeline
from rag_sec.observability import configure_observability, shutdown_observability
from rag_sec.providers import warmup_embedding_model


async def main() -> None:
    loaded = await FinanceBenchSuite().load()
    evaluation_cases = loaded.cases
    resolution = loaded.resolution

    print("FinanceBench cases:", len(resolution.cases))
    print("SEC-compatible cases:", len(evaluation_cases))
    print("Excluded cases:", len(resolution.cases) - len(evaluation_cases))
    print("Unsupported documents:", len(resolution.unsupported_documents))

    database = get_database_manager()
    await database.initialize()

    try:
        pipeline = IngestionPipeline()
        await asyncio.to_thread(
            warmup_embedding_model,
            pipeline.embeddings,
        )
        report = await prepare_financebench_corpus(
            pipeline=pipeline,
            cases=evaluation_cases,
        )

        print("\n=== FinanceBench Corpus ===")
        print("Requested filings:", report.requested_filings)
        print("Processed filings:", report.processed_filings)
        print("Skipped filings:", report.skipped_filings)
        print("Failed filings:", report.failed_filings)
        print(
            "Missing accession cases:",
            len(report.missing_accession_case_ids),
        )
        print("Corpus ready:", report.ready)

        if report.failed_accessions:
            print("\nFailed accessions:")
            for accession, error in report.failed_accessions.items():
                print(f"- {accession}: {error}")

        if not report.ready:
            raise RuntimeError(
                "FinanceBench corpus preparation did not complete successfully."
            )
    finally:
        await database.close()


if __name__ == "__main__":
    configure_observability()

    try:
        asyncio.run(main())
    finally:
        shutdown_observability()

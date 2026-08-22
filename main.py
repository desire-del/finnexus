import asyncio

from rag_sec.database.manager import (
    get_database_manager,
)

from rag_sec.ingestion.pipeline import (
    IngestionPipeline,
)


async def main():

    db = get_database_manager()

    await db.initialize()

    pipeline = IngestionPipeline()

    result = await pipeline.ingest_latest(
        "AAPL",
        form_type="10-K",
    )

    print("\n--- INGESTION RESULT ---")
    print(result)

    await db.close()


if __name__ == "__main__":
    asyncio.run(main())
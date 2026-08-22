import asyncio

from rag_sec.database.manager import (
    get_database_manager,
)
from rag_sec.generation.generator import Generator
from rag_sec.retrieval import Retriever


async def main():

    db = get_database_manager()

    await db.initialize()

    retriever = Retriever(
        top_k=5,
        dense_top_k=20,
        lexical_top_k=20,
    )

    generator = Generator()

    question = (
        "How did foreign exchange "
        "affect Apple's results?"
    )

    # ======================================
    # RETRIEVAL
    # ======================================

    documents = await retriever.search(
        question,
        ticker="AAPL",
        form_type="10-K",
    )

    # ======================================
    # GENERATION
    # ======================================

    result = await generator.generate(
        question,
        documents,
    )

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

        print(
            source.source_url
        )

        print()

    await db.close()


if __name__ == "__main__":
    asyncio.run(main())
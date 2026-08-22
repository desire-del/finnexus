import asyncio

from rag_sec.database.manager import (
    get_database_manager,
)
from rag_sec.retrieval import Retriever


async def main():

    db = get_database_manager()

    await db.initialize()

    retriever = Retriever(
        top_k=5
    )

    results = await retriever.search(
        (
            "What risks does Apple identify "
            "regarding cybersecurity?"
        ),
        ticker="AAPL",
        form_type="10-K",
    )

    for i, (
        document,
        distance,
    ) in enumerate(
        results,
        start=1,
    ):

        print(
            "\n"
            + "=" * 80
        )

        print(
            f"RESULT {i}"
        )

        print(
            f"Distance: "
            f"{float(distance):.4f}"
        )

        print(
            "Company:",
            document.metadata.get(
                "company_name"
            ),
        )

        print(
            "Ticker:",
            document.metadata.get(
                "ticker"
            ),
        )

        print(
            "Section:",
            document.metadata.get(
                "section"
            ),
        )

        print(
            "Item:",
            document.metadata.get(
                "item"
            ),
        )

        print(
            "Form:",
            document.metadata.get(
                "form_type"
            ),
        )

        print(
            "Accession:",
            document.metadata.get(
                "accession_number"
            ),
        )

        print("\nTEXT:\n")

        print(
            document.page_content[
                :1500
            ]
        )

    await db.close()


if __name__ == "__main__":
    asyncio.run(main())
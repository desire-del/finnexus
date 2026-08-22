from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)
from pydantic import BaseModel

from rag_sec.generation.chat_model import get_chat_model
from rag_sec.generation.context import ContextBuilder
from rag_sec.logging import get_logger
from rag_sec.prompts import (
    GENERATION_SYSTEM_PROMPT,
    build_generation_user_prompt,
)

log = get_logger(__name__)


class GeneratedAnswer(BaseModel):
    answer: str
    cited_source_ids: list[str]


class SourceInfo(BaseModel):
    source_id: str

    company_name: str | None = None
    ticker: str | None = None

    form_type: str | None = None
    filing_date: str | None = None

    accession_number: str | None = None

    section: str | None = None
    part: str | None = None
    item: str | None = None

    source_url: str | None = None


class RAGAnswer(BaseModel):
    answer: str
    sources: list[SourceInfo]


class Generator:

    def __init__(self):
        self.context_builder = ContextBuilder()
        self.model = (
            get_chat_model()
            .with_structured_output(
                GeneratedAnswer
            )
        )

    async def generate(
        self,
        question: str,
        documents,
    ) -> RAGAnswer:

        if not documents:
            return RAGAnswer(
                answer=(
                    "I could not find sufficient "
                    "evidence in the retrieved SEC filings "
                    "to answer this question."
                ),
                sources=[],
            )

        bundle = self.context_builder.build(
            documents
        )

        message = build_generation_user_prompt(
            question=question,
            context=bundle.text,
        )

        result = await self.model.ainvoke(
            [
                SystemMessage(
                    content=(
                        GENERATION_SYSTEM_PROMPT
                    )
                ),
                HumanMessage(
                    content=message
                ),
            ]
        )

        # -----------------------------------
        # Validate citations
        # -----------------------------------

        valid_source_ids = set(
            bundle.sources
        )

        cited_ids = [
            source_id
            for source_id
            in result.cited_source_ids
            if source_id in valid_source_ids
        ]

        sources = []

        for source_id in cited_ids:

            document = (
                bundle.sources[source_id]
            )

            metadata = (
                document.metadata
            )

            sources.append(
                SourceInfo(
                    source_id=source_id,

                    company_name=(
                        metadata.get(
                            "company_name"
                        )
                    ),

                    ticker=metadata.get(
                        "ticker"
                    ),

                    form_type=metadata.get(
                        "form_type"
                    ),

                    filing_date=str(
                        metadata.get(
                            "filing_date"
                        )
                    )
                    if metadata.get(
                        "filing_date"
                    )
                    else None,

                    accession_number=(
                        metadata.get(
                            "accession_number"
                        )
                    ),

                    section=metadata.get(
                        "section"
                    ),

                    part=metadata.get(
                        "part"
                    ),

                    item=metadata.get(
                        "item"
                    ),

                    source_url=metadata.get(
                        "source_url"
                    ),
                )
            )

        log.info(
            "answer_generated",
            source_count=len(sources),
        )

        return RAGAnswer(
            answer=result.answer,
            sources=sources,
        )

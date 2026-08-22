from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)
from pydantic import BaseModel

from rag_sec.generation.context import ContextBuilder
from rag_sec.logging import get_logger
from rag_sec.observability import (
    Phase,
    set_span_attributes,
    set_span_input,
    set_span_output,
    track,
)
from rag_sec.prompts import (
    GENERATION_SYSTEM_PROMPT,
    build_generation_user_prompt,
)
from rag_sec.providers import get_chat_model

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

    @track(
        name="generation.answer",
        phase=Phase.GENERATION,
        tags=["component:generator"],
    )
    async def generate(
        self,
        question: str,
        documents,
    ) -> RAGAnswer:

        set_span_attributes(
            {
                "rag.generation.question_length": len(question),
                "rag.generation.document_count": len(documents),
                "rag.generation.model": (
                    self.model.__class__.__name__
                ),
            }
        )
        set_span_input(
            {
                "question_length": len(question),
                "document_count": len(documents),
            }
        )

        if not documents:
            result = RAGAnswer(
                answer=(
                    "I could not find sufficient "
                    "evidence in the retrieved SEC filings "
                    "to answer this question."
                ),
                sources=[],
            )

            set_span_output(
                {
                    "source_count": 0,
                    "answer_length": len(result.answer),
                }
            )

            return result

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

        set_span_attributes(
            {
                "rag.generation.source_count": len(sources),
                "rag.generation.answer_length": len(result.answer),
            }
        )
        set_span_output(
            {
                "source_count": len(sources),
                "answer_length": len(result.answer),
            }
        )

        return RAGAnswer(
            answer=result.answer,
            sources=sources,
        )

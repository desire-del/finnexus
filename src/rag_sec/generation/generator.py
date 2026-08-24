from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)

from rag_sec.generation.context import ContextBuilder
from rag_sec.generation.models import GeneratedAnswer, RAGAnswer
from rag_sec.generation.sources import build_sources
from rag_sec.generation.usage import extract_token_usage
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


class Generator:
    def __init__(self):
        self.context_builder = ContextBuilder()
        self.model = get_chat_model().with_structured_output(
            GeneratedAnswer,
            include_raw=True,
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
                "rag.generation.model": (self.model.__class__.__name__),
            }
        )
        set_span_input(
            {
                "question_length": len(question),
                "document_count": len(documents),
            }
        )

        if not documents:
            empty_result = RAGAnswer(
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
                    "answer_length": len(empty_result.answer),
                }
            )

            return empty_result

        bundle = self.context_builder.build(documents)

        message = build_generation_user_prompt(
            question=question,
            context=bundle.text,
        )

        messages = [
            SystemMessage(content=(GENERATION_SYSTEM_PROMPT)),
            HumanMessage(content=message),
        ]
        response = await self.model.ainvoke(messages)

        if isinstance(response, dict) and "parsed" in response:
            parsing_error = response.get("parsing_error")

            if parsing_error is not None:
                raise ValueError(
                    "The model response could not be parsed as a cited answer."
                ) from parsing_error

            parsed_result = response.get("parsed")
            raw_message = response.get("raw")
        else:
            parsed_result = response
            raw_message = None

        if not isinstance(parsed_result, GeneratedAnswer):
            parsed_result = GeneratedAnswer.model_validate(parsed_result)

        usage = extract_token_usage(raw_message, messages, parsed_result.answer)
        sources = build_sources(parsed_result.cited_source_ids, bundle.sources)

        log.info(
            "answer_generated",
            source_count=len(sources),
        )

        set_span_attributes(
            {
                "rag.generation.source_count": len(sources),
                "rag.generation.answer_length": len(parsed_result.answer),
                "llm.token_count.prompt": usage.input_tokens,
                "llm.token_count.completion": usage.output_tokens,
                "llm.token_count.total": usage.total_tokens,
                "rag.generation.token_usage_estimated": usage.estimated,
            }
        )
        set_span_output(
            {
                "source_count": len(sources),
                "answer_length": len(parsed_result.answer),
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "total_tokens": usage.total_tokens,
                "token_usage_estimated": usage.estimated,
            }
        )

        return RAGAnswer(
            answer=parsed_result.answer,
            sources=sources,
            usage=usage,
        )

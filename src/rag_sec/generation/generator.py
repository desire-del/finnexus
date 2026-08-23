import re
from urllib.parse import quote, urldefrag

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)
from pydantic import BaseModel, Field

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
    deep_link: str | None = None

    chunk_id: str | None = None
    chunk_index: int | None = None
    page: int | None = None
    token_count: int | None = None

    excerpt: str | None = None


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated: bool = False


class QueryMetrics(BaseModel):
    total_latency_ms: float = 0.0
    embedding_latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    generation_throughput_tokens_per_second: float = 0.0
    retrieval_throughput_documents_per_second: float = 0.0
    retrieved_documents: int = 0
    cited_sources: int = 0


class RAGAnswer(BaseModel):
    answer: str
    sources: list[SourceInfo]
    usage: TokenUsage = Field(default_factory=TokenUsage)
    metrics: QueryMetrics = Field(default_factory=QueryMetrics)


def normalized_excerpt(text: str, *, max_characters: int = 900) -> str:
    """Create a compact, readable excerpt while preserving source wording."""
    plain_text = re.sub(r"!\[([^]]*)]\([^)]+\)", r"\1", text)
    plain_text = re.sub(r"\[([^]]+)]\([^)]+\)", r"\1", plain_text)
    plain_text = re.sub(r"<[^>]+>", " ", plain_text)
    plain_text = re.sub(r"[*_`#>|]", " ", plain_text)
    normalized = re.sub(r"\s+", " ", plain_text).strip()

    if len(normalized) <= max_characters:
        return normalized

    truncated = normalized[:max_characters].rsplit(" ", 1)[0]
    return f"{truncated}…"


def build_source_deep_link(source_url: str | None, excerpt: str) -> str | None:
    """Link to and highlight the cited passage with a text fragment."""
    if not source_url:
        return None

    base_url, _fragment = urldefrag(source_url)
    target = normalized_excerpt(excerpt, max_characters=180)

    if not target:
        return base_url

    return f"{base_url}#:~:text={quote(target, safe='')}"


def estimate_token_count(text: str) -> int:
    """Estimate tokens only when the model provider omits usage metadata."""
    if not text:
        return 0

    return max(1, round(len(text) / 4))


def token_usage(raw_message, messages, answer: str) -> TokenUsage:
    usage = getattr(raw_message, "usage_metadata", None) or {}
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")

    if input_tokens is not None and output_tokens is not None:
        total_tokens = usage.get("total_tokens")
        return TokenUsage(
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
            total_tokens=int(total_tokens or input_tokens + output_tokens),
            estimated=False,
        )

    estimated_input = estimate_token_count(
        "\n".join(str(message.content) for message in messages)
    )
    estimated_output = estimate_token_count(answer)
    return TokenUsage(
        input_tokens=estimated_input,
        output_tokens=estimated_output,
        total_tokens=estimated_input + estimated_output,
        estimated=True,
    )


class Generator:

    def __init__(self):
        self.context_builder = ContextBuilder()
        self.model = (
            get_chat_model()
            .with_structured_output(
                GeneratedAnswer,
                include_raw=True,
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

        messages = [
            SystemMessage(
                content=(
                    GENERATION_SYSTEM_PROMPT
                )
            ),
            HumanMessage(
                content=message
            ),
        ]
        response = await self.model.ainvoke(messages)

        if isinstance(response, dict) and "parsed" in response:
            parsing_error = response.get("parsing_error")

            if parsing_error is not None:
                raise ValueError(
                    "The model response could not be parsed as a cited answer."
                ) from parsing_error

            result = response.get("parsed")
            raw_message = response.get("raw")
        else:
            result = response
            raw_message = None

        if not isinstance(result, GeneratedAnswer):
            result = GeneratedAnswer.model_validate(result)

        usage = token_usage(raw_message, messages, result.answer)

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
            excerpt = normalized_excerpt(
                document.page_content
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

                    deep_link=build_source_deep_link(
                        metadata.get("source_url"),
                        document.page_content,
                    ),

                    chunk_id=metadata.get(
                        "chunk_id"
                    ),

                    chunk_index=metadata.get(
                        "chunk_index"
                    ),

                    page=metadata.get(
                        "page"
                    ),

                    token_count=metadata.get(
                        "token_count"
                    ),

                    excerpt=excerpt,
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
                "llm.token_count.prompt": usage.input_tokens,
                "llm.token_count.completion": usage.output_tokens,
                "llm.token_count.total": usage.total_tokens,
                "rag.generation.token_usage_estimated": usage.estimated,
            }
        )
        set_span_output(
            {
                "source_count": len(sources),
                "answer_length": len(result.answer),
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "total_tokens": usage.total_tokens,
                "token_usage_estimated": usage.estimated,
            }
        )

        return RAGAnswer(
            answer=result.answer,
            sources=sources,
            usage=usage,
        )

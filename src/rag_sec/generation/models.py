from pydantic import BaseModel, Field


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

import re
from urllib.parse import quote, urldefrag

from langchain_core.documents import Document

from rag_sec.generation.models import SourceInfo


def normalized_excerpt(text: str, *, max_characters: int = 900) -> str:
    plain_text = re.sub(r"!\[([^]]*)]\([^)]+\)", r"\1", text)
    plain_text = re.sub(r"\[([^]]+)]\([^)]+\)", r"\1", plain_text)
    plain_text = re.sub(r"<[^>]+>", " ", plain_text)
    plain_text = re.sub(r"[*_`#>|]", " ", plain_text)
    normalized = re.sub(r"\s+", " ", plain_text).strip()
    if len(normalized) <= max_characters:
        return normalized
    return f"{normalized[:max_characters].rsplit(' ', 1)[0]}…"


def build_source_deep_link(source_url: str | None, excerpt: str) -> str | None:
    if not source_url:
        return None
    base_url, _ = urldefrag(source_url)
    target = normalized_excerpt(excerpt, max_characters=180)
    return f"{base_url}#:~:text={quote(target, safe='')}" if target else base_url


def build_sources(
    cited_source_ids: list[str],
    available_sources: dict[str, Document],
) -> list[SourceInfo]:
    sources = []
    for source_id in dict.fromkeys(cited_source_ids):
        document = available_sources.get(source_id)
        if document is None:
            continue
        metadata = document.metadata
        sources.append(
            SourceInfo(
                source_id=source_id,
                company_name=metadata.get("company_name"),
                ticker=metadata.get("ticker"),
                form_type=metadata.get("form_type"),
                filing_date=(
                    str(metadata["filing_date"])
                    if metadata.get("filing_date")
                    else None
                ),
                accession_number=metadata.get("accession_number"),
                section=metadata.get("section"),
                part=metadata.get("part"),
                item=metadata.get("item"),
                source_url=metadata.get("source_url"),
                deep_link=build_source_deep_link(
                    metadata.get("source_url"), document.page_content
                ),
                chunk_id=metadata.get("chunk_id"),
                chunk_index=metadata.get("chunk_index"),
                page=metadata.get("page"),
                token_count=metadata.get("token_count"),
                excerpt=normalized_excerpt(document.page_content),
            )
        )
    return sources

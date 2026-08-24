from dataclasses import dataclass

from langchain_core.documents import Document

from rag_sec.observability import (
    Phase,
    set_span_attributes,
    set_span_input,
    set_span_output,
    track,
)


@dataclass
class ContextBundle:
    text: str
    sources: dict[str, Document]


class ContextBuilder:
    @track(
        name="generation.build_context",
        phase=Phase.GENERATION,
        tags=["component:context"],
    )
    def build(
        self,
        documents: list[Document],
    ) -> ContextBundle:

        set_span_input(
            {
                "document_count": len(documents),
            }
        )

        sources: dict[str, Document] = {}
        blocks = []

        seen_chunks = set()

        for document in documents:
            chunk_id = document.metadata.get("chunk_id")

            # Avoid duplicate retrieved chunks
            if chunk_id in seen_chunks:
                continue

            if chunk_id:
                seen_chunks.add(chunk_id)

            source_id = f"S{len(sources) + 1}"

            sources[source_id] = document

            metadata = document.metadata

            block = f"""
[{source_id}]
Company: {metadata.get("company_name")}
Ticker: {metadata.get("ticker")}
Form: {metadata.get("form_type")}
Filing date: {metadata.get("filing_date")}
Accession number: {metadata.get("accession_number")}
Section: {metadata.get("section")}
Part: {metadata.get("part")}
Item: {metadata.get("item")}
Source URL: {metadata.get("source_url")}

{document.page_content}
""".strip()

            blocks.append(block)

        bundle = ContextBundle(
            text="\n\n---\n\n".join(blocks),
            sources=sources,
        )

        set_span_attributes(
            {
                "rag.context.document_count": len(documents),
                "rag.context.source_count": len(sources),
                "rag.context.character_count": len(bundle.text),
            }
        )
        set_span_output(
            {
                "source_count": len(sources),
                "character_count": len(bundle.text),
            }
        )

        return bundle

# src/rag_sec/ingestion/chunker.py

import hashlib
from uuid import UUID

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_sec.schemas.chunk import ChunkDraft, ChunkLocator
from rag_sec.schemas.filing import FilingSection


class SectionChunker:

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splitter = (
            RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name="cl100k_base",
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=[
                    "\n\n",
                    "\n",
                    ". ",
                    " ",
                    "",
                ],
            )
        )

    def chunk(
        self,
        *,
        sections: list[FilingSection],
        filing_id: UUID,
        processing_version_id: UUID,
        source_url: str,
    ) -> list[ChunkDraft]:

        chunks = []
        chunk_index = 0

        for section in sections:

            texts = self.splitter.split_text(
                section.content
            )

            for text in texts:

                text = text.strip()

                if not text:
                    continue

                content_hash = hashlib.sha256(
                    text.encode("utf-8")
                ).hexdigest()

                chunk_id = hashlib.sha256(
                    (
                        f"{processing_version_id}:"
                        f"{chunk_index}:"
                        f"{content_hash}"
                    ).encode("utf-8")
                ).hexdigest()

                chunks.append(
                    ChunkDraft(
                        chunk_id=chunk_id,
                        filing_id=filing_id,
                        processing_version_id=(
                            processing_version_id
                        ),
                        chunk_index=chunk_index,
                        text=text,
                        content_hash=content_hash,

                        locator=ChunkLocator(
                            section=section.name,
                            part=section.part,
                            item=section.item,
                            source_url=source_url,
                        ),

                        heading_path=[
                            value
                            for value in [
                                section.part,
                                section.item,
                                section.name,
                            ]
                            if value
                        ],

                        metadata={
                            "section_warnings":
                                section.warnings
                        },
                    )
                )

                chunk_index += 1

        return chunks
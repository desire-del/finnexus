import hashlib
import re
from uuid import UUID

import tiktoken

from rag_sec.schemas.chunk import (
    ChunkDraft,
    ChunkLocator,
)
from rag_sec.schemas.filing import FilingSection


class SectionChunker:
    """
    Structure-aware chunker for SEC filings.

    Strategy:
    1. Preserve SEC sections.
    2. Split sections into Markdown blocks.
    3. Accumulate blocks until max_tokens.
    4. Apply a small overlap between chunks.
    """

    def __init__(
        self,
        max_tokens: int = 800,
        overlap_tokens: int = 100,
        encoding_name: str = "cl100k_base",
    ):
        if overlap_tokens >= max_tokens:
            raise ValueError(
                "overlap_tokens must be smaller "
                "than max_tokens."
            )

        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

        self.encoding = tiktoken.get_encoding(
            encoding_name
        )

    def chunk(
        self,
        *,
        sections: list[FilingSection],
        filing_id: UUID,
        processing_version_id: UUID,
        source_url: str,
    ) -> list[ChunkDraft]:

        chunks: list[ChunkDraft] = []

        chunk_index = 0

        for section in sections:

            section_chunks = self._chunk_section(
                section.content
            )

            for text in section_chunks:

                text = self._normalize(text)

                if not text:
                    continue

                content_hash = self._hash(text)

                chunk_id = self._build_chunk_id(
                    processing_version_id=(
                        processing_version_id
                    ),
                    chunk_index=chunk_index,
                    content_hash=content_hash,
                )

                chunk = ChunkDraft(
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

                    heading_path=self._heading_path(
                        section
                    ),

                    token_count=self._count_tokens(
                        text
                    ),

                    metadata={
                        "section_warnings":
                            section.warnings
                    },
                )

                chunks.append(chunk)

                chunk_index += 1

        return chunks

    def _chunk_section(
        self,
        text: str,
    ) -> list[str]:
        """
        Split one SEC section while preserving
        Markdown paragraphs/tables when possible.
        """

        text = self._normalize(text)

        blocks = self._split_blocks(text)

        chunks: list[str] = []

        current_blocks: list[str] = []
        current_tokens = 0

        for block in blocks:

            block_tokens = self._count_tokens(
                block
            )

            # Huge single block:
            # fallback to token-window splitting
            if block_tokens > self.max_tokens:

                if current_blocks:
                    chunks.append(
                        "\n\n".join(
                            current_blocks
                        )
                    )

                    current_blocks = []
                    current_tokens = 0

                chunks.extend(
                    self._split_large_block(
                        block
                    )
                )

                continue

            # Block fits current chunk
            if (
                current_tokens + block_tokens
                <= self.max_tokens
            ):
                current_blocks.append(block)
                current_tokens += block_tokens

                continue

            # Finish current chunk
            if current_blocks:

                current_text = "\n\n".join(
                    current_blocks
                )

                chunks.append(current_text)

                overlap = self._get_overlap_blocks(
                    current_blocks
                )

                current_blocks = overlap

                current_tokens = sum(
                    self._count_tokens(b)
                    for b in current_blocks
                )

            current_blocks.append(block)
            current_tokens += block_tokens

        if current_blocks:
            chunks.append(
                "\n\n".join(
                    current_blocks
                )
            )

        return chunks

    def _split_blocks(
        self,
        text: str,
    ) -> list[str]:
        """
        Split on blank lines.

        This tends to preserve:
        - paragraphs
        - headings
        - lists
        - Markdown tables
        """

        blocks = re.split(
            r"\n\s*\n",
            text,
        )

        return [
            block.strip()
            for block in blocks
            if block.strip()
        ]

    def _split_large_block(
        self,
        text: str,
    ) -> list[str]:

        tokens = self.encoding.encode(text)

        step = (
            self.max_tokens
            - self.overlap_tokens
        )

        chunks = []

        for start in range(
            0,
            len(tokens),
            step,
        ):

            token_slice = tokens[
                start:
                start + self.max_tokens
            ]

            if not token_slice:
                break

            chunk = self.encoding.decode(
                token_slice
            ).strip()

            if chunk:
                chunks.append(chunk)

            if (
                start + self.max_tokens
                >= len(tokens)
            ):
                break

        return chunks

    def _get_overlap_blocks(
        self,
        blocks: list[str],
    ) -> list[str]:

        overlap: list[str] = []

        token_count = 0

        for block in reversed(blocks):

            block_tokens = self._count_tokens(
                block
            )

            if (
                token_count + block_tokens
                > self.overlap_tokens
            ):
                break

            overlap.insert(0, block)

            token_count += block_tokens

        return overlap

    def _count_tokens(
        self,
        text: str,
    ) -> int:

        return len(
            self.encoding.encode(text)
        )

    @staticmethod
    def _normalize(
        text: str,
    ) -> str:
        """
        Conservative normalization.

        Do not destroy SEC structure, numbers
        or Markdown tables.
        """

        text = text.replace(
            "\r\n",
            "\n",
        )

        text = text.replace(
            "\r",
            "\n",
        )

        lines = [
            line.rstrip()
            for line in text.splitlines()
        ]

        text = "\n".join(lines)

        # Maximum two consecutive blank lines
        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text,
        )

        return text.strip()

    @staticmethod
    def _hash(
        text: str,
    ) -> str:

        return hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _build_chunk_id(
        *,
        processing_version_id: UUID,
        chunk_index: int,
        content_hash: str,
    ) -> str:

        value = (
            f"{processing_version_id}:"
            f"{chunk_index}:"
            f"{content_hash}"
        )

        return hashlib.sha256(
            value.encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _heading_path(
        section: FilingSection,
    ) -> list[str]:

        headings = []

        if section.part:
            headings.append(section.part)

        if section.item:
            headings.append(section.item)

        if section.name:
            headings.append(section.name)

        return headings
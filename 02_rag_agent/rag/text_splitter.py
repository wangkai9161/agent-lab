from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rag.document_loader import Document


@dataclass(frozen=True)
class TextChunk:
    """A retrievable document fragment with source metadata."""

    content: str
    source: str
    chunk_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


class ChineseTextSplitter:
    """Split Chinese experiment documents by paragraphs and length."""

    def __init__(
        self,
        max_chars: int = 500,
        overlap_chars: int = 80,
    ) -> None:
        if max_chars <= 0:
            raise ValueError("max_chars 必须大于 0")

        if overlap_chars < 0:
            raise ValueError("overlap_chars 不能小于 0")

        if overlap_chars >= max_chars:
            raise ValueError("overlap_chars 必须小于 max_chars")

        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    def split_documents(
        self,
        documents: list[Document],
    ) -> list[TextChunk]:
        chunks: list[TextChunk] = []

        for document in documents:
            chunks.extend(self.split_document(document))

        return chunks

    def split_document(self, document: Document) -> list[TextChunk]:
        paragraphs = self._split_paragraphs(document.content)
        chunks: list[TextChunk] = []
        current_parts: list[str] = []
        current_start_line = 1
        chunk_index = 0

        for paragraph, line_start, line_end in paragraphs:
            if not current_parts:
                current_start_line = line_start

            candidate = "\n\n".join([*current_parts, paragraph])

            if len(candidate) <= self.max_chars:
                current_parts.append(paragraph)
                continue

            if current_parts:
                chunk_index += 1
                content = "\n\n".join(current_parts)
                chunks.append(
                    self._make_chunk(
                        document=document,
                        content=content,
                        chunk_index=chunk_index,
                        line_start=current_start_line,
                        line_end=line_start - 1,
                    )
                )
                current_parts = [self._build_overlap(content), paragraph]
                current_start_line = line_start
            else:
                for piece in self._split_long_text(paragraph):
                    chunk_index += 1
                    chunks.append(
                        self._make_chunk(
                            document=document,
                            content=piece,
                            chunk_index=chunk_index,
                            line_start=line_start,
                            line_end=line_end,
                        )
                    )
                current_parts = []

        if current_parts:
            chunk_index += 1
            chunks.append(
                self._make_chunk(
                    document=document,
                    content="\n\n".join(current_parts),
                    chunk_index=chunk_index,
                    line_start=current_start_line,
                    line_end=document.metadata.get("line_end", current_start_line),
                )
            )

        return [chunk for chunk in chunks if chunk.content.strip()]

    def _make_chunk(
        self,
        document: Document,
        content: str,
        chunk_index: int,
        line_start: int,
        line_end: int,
    ) -> TextChunk:
        return TextChunk(
            content=content.strip(),
            source=document.source,
            chunk_id=f"{document.metadata.get('file_name', 'document')}-{chunk_index}",
            metadata={
                **document.metadata,
                "chunk_index": chunk_index,
                "line_start": max(1, line_start),
                "line_end": max(line_start, line_end),
            },
        )

    def _build_overlap(self, content: str) -> str:
        if self.overlap_chars == 0:
            return ""

        return content[-self.overlap_chars :].strip()

    def _split_long_text(self, text: str) -> list[str]:
        step = self.max_chars - self.overlap_chars
        pieces = []

        for start in range(0, len(text), step):
            piece = text[start : start + self.max_chars].strip()
            if piece:
                pieces.append(piece)

        return pieces

    def _split_paragraphs(self, text: str) -> list[tuple[str, int, int]]:
        paragraphs: list[tuple[str, int, int]] = []
        buffer: list[str] = []
        start_line = 1

        for line_number, raw_line in enumerate(text.splitlines(), start=1):
            line = raw_line.strip()

            if not line:
                if buffer:
                    paragraphs.append(
                        ("\n".join(buffer), start_line, line_number - 1)
                    )
                    buffer = []
                continue

            if not buffer:
                start_line = line_number

            buffer.append(line)

        if buffer:
            paragraphs.append(
                ("\n".join(buffer), start_line, start_line + len(buffer) - 1)
            )

        return paragraphs

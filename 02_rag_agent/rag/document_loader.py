from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SUPPORTED_SUFFIXES = {".txt", ".md", ".log"}


@dataclass(frozen=True)
class Document:
    """A normalized text document loaded from local files."""

    content: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


class DocumentLoader:
    """Load local experiment documents into a common Document format."""

    def __init__(
        self,
        supported_suffixes: set[str] | None = None,
        encoding: str = "utf-8",
    ) -> None:
        self.supported_suffixes = supported_suffixes or SUPPORTED_SUFFIXES
        self.encoding = encoding

    def load_file(self, path: str | Path) -> Document:
        file_path = Path(path)

        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在：{file_path}")

        if not file_path.is_file():
            raise ValueError(f"路径不是文件：{file_path}")

        suffix = file_path.suffix.lower()
        if suffix not in self.supported_suffixes:
            raise ValueError(
                f"暂不支持的文件类型：{suffix}，"
                f"当前支持：{sorted(self.supported_suffixes)}"
            )

        content = file_path.read_text(encoding=self.encoding)
        line_count = len(content.splitlines())

        return Document(
            content=content,
            source=str(file_path),
            metadata={
                "file_name": file_path.name,
                "suffix": suffix,
                "line_start": 1,
                "line_end": line_count,
            },
        )

    def load_directory(
        self,
        directory: str | Path,
        recursive: bool = True,
    ) -> list[Document]:
        directory_path = Path(directory)

        if not directory_path.exists():
            raise FileNotFoundError(f"目录不存在：{directory_path}")

        if not directory_path.is_dir():
            raise ValueError(f"路径不是目录：{directory_path}")

        pattern = "**/*" if recursive else "*"
        documents: list[Document] = []

        for file_path in sorted(directory_path.glob(pattern)):
            if (
                file_path.is_file()
                and file_path.suffix.lower() in self.supported_suffixes
            ):
                documents.append(self.load_file(file_path))

        return documents

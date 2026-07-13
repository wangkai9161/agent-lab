from pathlib import Path

import pytest

from rag.document_loader import DocumentLoader
from rag.text_splitter import ChineseTextSplitter


def test_load_supported_text_file(tmp_path: Path) -> None:
    note = tmp_path / "experiment.md"
    note.write_text(
        "# 实验记录\n\n训练 UNet 时出现 CUDA out of memory。",
        encoding="utf-8",
    )

    loader = DocumentLoader()
    document = loader.load_file(note)

    assert "CUDA out of memory" in document.content
    assert document.metadata["file_name"] == "experiment.md"
    assert document.metadata["line_start"] == 1


def test_load_directory_ignores_unsupported_files(tmp_path: Path) -> None:
    (tmp_path / "note.md").write_text("有效资料", encoding="utf-8")
    (tmp_path / "image.png").write_text("ignore", encoding="utf-8")

    documents = DocumentLoader().load_directory(tmp_path)

    assert len(documents) == 1
    assert documents[0].metadata["file_name"] == "note.md"


def test_reject_unsupported_file_type(tmp_path: Path) -> None:
    data = tmp_path / "data.csv"
    data.write_text("a,b", encoding="utf-8")

    with pytest.raises(ValueError):
        DocumentLoader().load_file(data)


def test_splitter_keeps_source_metadata(tmp_path: Path) -> None:
    note = tmp_path / "note.md"
    note.write_text(
        "第一段：训练配置。\n\n第二段：显存不足时降低 batch size。",
        encoding="utf-8",
    )

    document = DocumentLoader().load_file(note)
    chunks = ChineseTextSplitter(max_chars=30, overlap_chars=5).split_document(
        document
    )

    assert chunks
    assert chunks[0].source == str(note)
    assert "chunk_index" in chunks[0].metadata

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag.document_loader import DocumentLoader
from rag.llm import AnswerGenerator, TemplateAnswerGenerator
from rag.text_splitter import ChineseTextSplitter
from rag.vector_store import InMemoryHybridVectorStore, SearchResult


@dataclass(frozen=True)
class RagResponse:
    question: str
    answer: str
    sources: list[dict[str, Any]]
    refused: bool = False

    def to_text(self) -> str:
        if self.refused:
            return self.answer

        evidence = []
        for index, source in enumerate(self.sources, start=1):
            evidence.append(
                f"{index}. {source['file_name']} / {source['chunk_id']} "
                f"行 {source['line_start']}-{source['line_end']}，"
                f"相似度 {source['score']:.2f}"
            )

        return f"已检索依据：\n" + "\n".join(evidence) + f"\n\n回答：\n{self.answer}"


class ExperimentRagAgent:
    """A traceable Chinese RAG agent for experiment documents."""

    def __init__(
        self,
        retriever: InMemoryHybridVectorStore | None = None,
        splitter: ChineseTextSplitter | None = None,
        loader: DocumentLoader | None = None,
        answer_generator: AnswerGenerator | None = None,
        min_confidence: float = 0.12,
    ) -> None:
        self.loader = loader or DocumentLoader()
        self.splitter = splitter or ChineseTextSplitter()
        self.retriever = retriever or InMemoryHybridVectorStore()
        self.answer_generator = answer_generator or TemplateAnswerGenerator()
        self.min_confidence = min_confidence

    def ingest_directory(self, directory: str | Path) -> int:
        documents = self.loader.load_directory(directory)
        chunks = self.splitter.split_documents(documents)
        self.retriever.add_chunks(chunks)
        return len(chunks)

    def ingest_files(self, paths: list[str | Path]) -> int:
        documents = [self.loader.load_file(path) for path in paths]
        chunks = self.splitter.split_documents(documents)
        self.retriever.add_chunks(chunks)
        return len(chunks)

    def save_index(self, path: str | Path) -> None:
        self.retriever.save(path)

    def load_index(self, path: str | Path) -> int:
        return self.retriever.load(path)

    def answer(self, question: str, top_k: int = 4) -> str:
        return self.answer_with_sources(question, top_k=top_k).to_text()

    def answer_with_sources(self, question: str, top_k: int = 4) -> RagResponse:
        results = self.retriever.search(
            query=question,
            top_k=top_k,
            min_score=self.min_confidence,
        )

        if not results:
            return RagResponse(
                question=question,
                answer=(
                "当前知识库没有检索到足够依据，不能确认原因。\n"
                    "建议补充训练日志、实验记录、论文笔记或项目 README 后再提问。"
                ),
                sources=[],
                refused=True,
            )

        return RagResponse(
            question=question,
            answer=self.answer_generator.generate(question, results),
            sources=[self._source_payload(result) for result in results],
        )

    def _format_evidence(self, results: list[SearchResult]) -> str:
        lines = []

        for index, result in enumerate(results, start=1):
            source_name = Path(result.source).name
            line_start = result.metadata.get("line_start", "?")
            line_end = result.metadata.get("line_end", "?")
            lines.append(
                f"{index}. {source_name} / {result.chunk_id} "
                f"行 {line_start}-{line_end}，相似度 {result.score:.2f}"
            )

        return "\n".join(lines)

    def _source_payload(self, result: SearchResult) -> dict[str, Any]:
        return {
            "source": result.source,
            "file_name": Path(result.source).name,
            "chunk_id": result.chunk_id,
            "score": result.score,
            "line_start": result.metadata.get("line_start", "?"),
            "line_end": result.metadata.get("line_end", "?"),
            "content": result.content,
        }

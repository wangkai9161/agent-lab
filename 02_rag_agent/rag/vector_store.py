from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rag.embedding_model import HashingEmbeddingModel
from rag.text_splitter import TextChunk


@dataclass(frozen=True)
class SearchResult:
    content: str
    source: str
    chunk_id: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class InMemoryHybridVectorStore:
    """In-memory hybrid retriever combining vector and keyword scores."""

    def __init__(
        self,
        embedding_model: HashingEmbeddingModel | None = None,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
    ) -> None:
        if vector_weight < 0 or keyword_weight < 0:
            raise ValueError("检索权重不能小于 0")

        total_weight = vector_weight + keyword_weight
        if total_weight == 0:
            raise ValueError("至少需要一个检索权重大于 0")

        self.embedding_model = embedding_model or HashingEmbeddingModel()
        self.vector_weight = vector_weight / total_weight
        self.keyword_weight = keyword_weight / total_weight
        self._chunks: list[TextChunk] = []
        self._vectors: list[list[float]] = []
        self._chunk_keys: set[str] = set()

    @property
    def size(self) -> int:
        return len(self._chunks)

    def add_chunks(self, chunks: list[TextChunk]) -> None:
        if not chunks:
            return

        new_chunks = [
            chunk for chunk in chunks if self._chunk_key(chunk) not in self._chunk_keys
        ]

        if not new_chunks:
            return

        self._chunks.extend(new_chunks)
        self._vectors.extend(
            self.embedding_model.embed_documents(
                [chunk.content for chunk in new_chunks]
            )
        )
        self._chunk_keys.update(self._chunk_key(chunk) for chunk in new_chunks)

    def clear(self) -> None:
        self._chunks = []
        self._vectors = []
        self._chunk_keys = set()

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "embedding": {
                "type": self.embedding_model.__class__.__name__,
                "dimension": self.embedding_model.dimension,
            },
            "weights": {
                "vector": self.vector_weight,
                "keyword": self.keyword_weight,
            },
            "items": [
                {
                    "chunk": {
                        "content": chunk.content,
                        "source": chunk.source,
                        "chunk_id": chunk.chunk_id,
                        "metadata": chunk.metadata,
                    },
                    "vector": vector,
                }
                for chunk, vector in zip(self._chunks, self._vectors)
            ],
        }

        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load(self, path: str | Path) -> int:
        input_path = Path(path)

        if not input_path.exists():
            raise FileNotFoundError(f"索引文件不存在：{input_path}")

        payload = json.loads(input_path.read_text(encoding="utf-8"))
        items = payload.get("items", [])

        self.clear()
        for item in items:
            chunk_payload = item["chunk"]
            chunk = TextChunk(
                content=chunk_payload["content"],
                source=chunk_payload["source"],
                chunk_id=chunk_payload["chunk_id"],
                metadata=chunk_payload.get("metadata", {}),
            )
            vector = item["vector"]
            if len(vector) != self.embedding_model.dimension:
                raise ValueError(
                    "索引向量维度与当前 embedding 模型不一致："
                    f"{len(vector)} != {self.embedding_model.dimension}"
                )
            self._chunks.append(chunk)
            self._vectors.append(vector)
            self._chunk_keys.add(self._chunk_key(chunk))

        return len(self._chunks)

    def search(
        self,
        query: str,
        top_k: int = 4,
        min_score: float = 0.05,
    ) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")

        if not self._chunks:
            return []

        query_vector = self.embedding_model.embed_query(query)
        query_terms = self._tokenize_keywords(query)
        scored_results: list[SearchResult] = []

        for chunk, chunk_vector in zip(self._chunks, self._vectors):
            vector_score = self._cosine_similarity(query_vector, chunk_vector)
            keyword_score = self._keyword_score(query_terms, chunk.content)
            score = (
                self.vector_weight * vector_score
                + self.keyword_weight * keyword_score
            )

            if score >= min_score:
                scored_results.append(
                    SearchResult(
                        content=chunk.content,
                        source=chunk.source,
                        chunk_id=chunk.chunk_id,
                        score=round(score, 4),
                        metadata=chunk.metadata,
                    )
                )

        return sorted(
            scored_results,
            key=lambda result: result.score,
            reverse=True,
        )[:top_k]

    def _cosine_similarity(
        self,
        left: list[float],
        right: list[float],
    ) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0

        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))

        if left_norm == 0 or right_norm == 0:
            return 0.0

        return dot / (left_norm * right_norm)

    def _keyword_score(
        self,
        query_terms: set[str],
        content: str,
    ) -> float:
        if not query_terms:
            return 0.0

        normalized_content = content.lower()
        hits = sum(1 for term in query_terms if term in normalized_content)
        return hits / len(query_terms)

    def _tokenize_keywords(self, text: str) -> set[str]:
        normalized = text.lower()
        terms = set(re.findall(r"[a-z0-9_./:-]+", normalized))
        terms.update(re.findall(r"[\u4e00-\u9fff]{2,}", normalized))
        return {term for term in terms if len(term) >= 2}

    def _chunk_key(self, chunk: TextChunk) -> str:
        digest = hashlib.md5(chunk.content.encode("utf-8")).hexdigest()
        return f"{chunk.source}:{chunk.chunk_id}:{digest}"

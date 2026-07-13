from __future__ import annotations

import hashlib
import math
import re
from collections import Counter


class HashingEmbeddingModel:
    """
    Lightweight local embedding model for deterministic tests and demos.

    It uses hashed token counts with L2 normalization. This is not a semantic
    model, but it provides stable retrieval behavior without network calls.
    """

    def __init__(self, dimension: int = 2048) -> None:
        if dimension <= 0:
            raise ValueError("dimension 必须大于 0")

        self.dimension = dimension

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = self._tokenize(text)

        if not tokens:
            return vector

        token_counts = Counter(tokens)

        for token, count in token_counts.items():
            index = self._hash_token(token) % self.dimension
            vector[index] += float(count)

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector

        return [value / norm for value in vector]

    def _hash_token(self, token: str) -> int:
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        return int(digest, 16)

    def _tokenize(self, text: str) -> list[str]:
        normalized = text.lower()
        words = re.findall(r"[a-z0-9_./:-]+", normalized)
        chinese_chars = re.findall(r"[\u4e00-\u9fff]", normalized)

        chinese_bigrams = [
            "".join(chinese_chars[index : index + 2])
            for index in range(len(chinese_chars) - 1)
        ]
        chinese_trigrams = [
            "".join(chinese_chars[index : index + 3])
            for index in range(len(chinese_chars) - 2)
        ]

        return words + chinese_bigrams + chinese_trigrams

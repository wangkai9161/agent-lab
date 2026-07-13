from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RagSettings:
    docs_dir: Path = Path("data/raw")
    index_path: Path = Path("vector_store/index.json")
    top_k: int = 4
    min_confidence: float = 0.12
    llm_provider: str = "template"
    llm_model: str = "qwen-plus"
    llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    llm_api_key: str | None = None


def load_settings() -> RagSettings:
    load_env_file(Path(".env"))

    return RagSettings(
        docs_dir=Path(os.getenv("RAG_DOCS_DIR", "data/raw")),
        index_path=Path(os.getenv("RAG_INDEX_PATH", "vector_store/index.json")),
        top_k=int(os.getenv("RAG_TOP_K", "4")),
        min_confidence=float(os.getenv("RAG_MIN_CONFIDENCE", "0.12")),
        llm_provider=os.getenv("RAG_LLM_PROVIDER", "template"),
        llm_model=os.getenv("DASHSCOPE_MODEL", os.getenv("RAG_LLM_MODEL", "qwen-plus")),
        llm_base_url=os.getenv(
            "DASHSCOPE_BASE_URL",
            os.getenv(
                "RAG_LLM_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
        ),
        llm_api_key=os.getenv("DASHSCOPE_API_KEY") or os.getenv("RAG_LLM_API_KEY"),
    )


def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key and key not in os.environ:
            os.environ[key] = value

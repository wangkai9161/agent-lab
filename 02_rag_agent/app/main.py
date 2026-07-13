from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag.config import load_settings
from rag.llm import build_answer_generator
from rag.rag_agent import ExperimentRagAgent


class IngestRequest(BaseModel):
    docs_dir: str | None = Field(default=None, description="资料目录")
    rebuild: bool = Field(default=False, description="是否强制重建索引")


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=4, ge=1, le=20)
    min_score: float | None = Field(default=None, ge=0)


class AnswerRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=4, ge=1, le=20)


settings = load_settings()
answer_generator = build_answer_generator(
    provider=settings.llm_provider,
    api_key=settings.llm_api_key,
    base_url=settings.llm_base_url,
    model=settings.llm_model,
)
agent = ExperimentRagAgent(
    answer_generator=answer_generator,
    min_confidence=settings.min_confidence,
)

app = FastAPI(
    title="Chinese Experiment RAG Agent",
    version="0.2.0",
    description="面向深度学习实验资料的可追溯 RAG 服务",
)


@app.on_event("startup")
def load_or_build_index() -> None:
    index_path = settings.index_path
    try:
        if index_path.exists():
            agent.load_index(index_path)
        else:
            agent.ingest_directory(settings.docs_dir)
            agent.save_index(index_path)
    except Exception:
        # Keep service bootable so /ingest can report and repair bad paths.
        return


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "chunks": agent.retriever.size,
        "docs_dir": str(settings.docs_dir),
        "index_path": str(settings.index_path),
        "llm_provider": settings.llm_provider,
    }


@app.post("/ingest")
def ingest(request: IngestRequest) -> dict[str, object]:
    docs_dir = Path(request.docs_dir) if request.docs_dir else settings.docs_dir

    try:
        if request.rebuild:
            agent.retriever.clear()
        chunk_count = agent.ingest_directory(docs_dir)
        agent.save_index(settings.index_path)
    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"{type(error).__name__}: {error}",
        ) from error

    return {
        "docs_dir": str(docs_dir),
        "index_path": str(settings.index_path),
        "chunks": agent.retriever.size,
        "new_chunks": chunk_count,
    }


@app.post("/search")
def search(request: SearchRequest) -> dict[str, object]:
    min_score = (
        settings.min_confidence
        if request.min_score is None
        else request.min_score
    )
    results = agent.retriever.search(
        request.query,
        top_k=request.top_k,
        min_score=min_score,
    )

    return {
        "query": request.query,
        "results": [
            {
                "source": result.source,
                "chunk_id": result.chunk_id,
                "score": result.score,
                "line_start": result.metadata.get("line_start"),
                "line_end": result.metadata.get("line_end"),
                "content": result.content,
            }
            for result in results
        ],
    }


@app.post("/answer")
def answer(request: AnswerRequest) -> dict[str, object]:
    response = agent.answer_with_sources(
        request.question,
        top_k=request.top_k,
    )
    return {
        "question": response.question,
        "answer": response.answer,
        "sources": response.sources,
        "refused": response.refused,
    }

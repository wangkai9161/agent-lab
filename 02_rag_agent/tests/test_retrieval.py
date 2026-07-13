from pathlib import Path

from rag.document_loader import Document
from rag.rag_agent import ExperimentRagAgent
from rag.text_splitter import ChineseTextSplitter
from rag.vector_store import InMemoryHybridVectorStore


def build_store() -> InMemoryHybridVectorStore:
    documents = [
        Document(
            content=(
                "训练日志显示 RuntimeError: CUDA out of memory。"
                "建议降低 batch size 或减小输入分辨率。"
            ),
            source="cuda_oom_note.md",
            metadata={"file_name": "cuda_oom_note.md", "line_start": 1, "line_end": 2},
        ),
        Document(
            content="UNet 模型适合医学图像分割，常用 dice loss 评估。",
            source="unet_note.md",
            metadata={"file_name": "unet_note.md", "line_start": 1, "line_end": 1},
        ),
    ]
    chunks = ChineseTextSplitter(max_chars=120).split_documents(documents)
    store = InMemoryHybridVectorStore()
    store.add_chunks(chunks)
    return store


def test_hybrid_retrieval_finds_cuda_oom_chunk() -> None:
    store = build_store()

    results = store.search("CUDA out of memory 怎么处理", top_k=2)

    assert results
    assert "batch size" in results[0].content
    assert results[0].score > 0


def test_store_deduplicates_same_chunks() -> None:
    store = build_store()
    original_size = store.size
    documents = [
        Document(
            content=(
                "训练日志显示 RuntimeError: CUDA out of memory。"
                "建议降低 batch size 或减小输入分辨率。"
            ),
            source="cuda_oom_note.md",
            metadata={"file_name": "cuda_oom_note.md", "line_start": 1, "line_end": 2},
        )
    ]
    chunks = ChineseTextSplitter(max_chars=120).split_documents(documents)

    store.add_chunks(chunks)

    assert store.size == original_size


def test_store_can_save_and_load_index(tmp_path: Path) -> None:
    store = build_store()
    index_path = tmp_path / "index.json"
    store.save(index_path)
    loaded_store = InMemoryHybridVectorStore()

    loaded_count = loaded_store.load(index_path)
    results = loaded_store.search("CUDA out of memory 怎么处理", top_k=1)

    assert loaded_count == store.size
    assert results
    assert "batch size" in results[0].content


def test_retrieval_returns_empty_for_unrelated_question() -> None:
    store = build_store()

    results = store.search("今天上海天气怎么样", top_k=2, min_score=0.5)

    assert results == []


def test_rag_agent_refuses_without_evidence() -> None:
    agent = ExperimentRagAgent(retriever=build_store(), min_confidence=0.5)

    answer = agent.answer("怎么配置 Kubernetes 集群？")

    assert "没有检索到足够依据" in answer


def test_rag_agent_returns_evidence() -> None:
    agent = ExperimentRagAgent(retriever=build_store())

    answer = agent.answer("训练出现 CUDA out of memory 的原因是什么？")

    assert "已检索依据" in answer
    assert "cuda_oom_note.md" in answer

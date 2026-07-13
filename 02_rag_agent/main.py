from __future__ import annotations

import argparse
from pathlib import Path

from rag.config import load_settings
from rag.llm import build_answer_generator
from rag.rag_agent import ExperimentRagAgent


def parse_args() -> argparse.Namespace:
    settings = load_settings()
    parser = argparse.ArgumentParser(
        description="中文实验资料 RAG Agent"
    )
    parser.add_argument(
        "--docs",
        default=str(settings.docs_dir),
        help="要加载的资料目录，默认 data/raw",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=settings.top_k,
        help="每次检索返回的片段数量",
    )
    parser.add_argument(
        "--index-path",
        default=str(settings.index_path),
        help="本地检索索引路径，默认 vector_store/index.json",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="忽略已有索引，重新加载资料并保存索引",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=settings.min_confidence,
        help="最低检索置信度，低于该分数时拒答",
    )
    parser.add_argument(
        "--llm-provider",
        default=settings.llm_provider,
        choices=["template", "qwen", "openai", "dashscope"],
        help="回答生成方式，默认 template，本地离线可用",
    )
    parser.add_argument(
        "--llm-model",
        default=settings.llm_model,
        help="OpenAI-compatible 模型名，默认 qwen-plus",
    )
    parser.add_argument(
        "--llm-base-url",
        default=settings.llm_base_url,
        help="OpenAI-compatible base_url",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings()
    docs_dir = Path(args.docs)
    index_path = Path(args.index_path)

    try:
        answer_generator = build_answer_generator(
            provider=args.llm_provider,
            api_key=settings.llm_api_key,
            base_url=args.llm_base_url,
            model=args.llm_model,
        )
    except Exception as error:
        print(f"回答模型初始化失败：{type(error).__name__}: {error}")
        return

    agent = ExperimentRagAgent(
        answer_generator=answer_generator,
        min_confidence=args.min_confidence,
    )

    print("=" * 60)
    print("中文实验资料 RAG Agent")
    print("输入 exit、quit 或 退出即可结束")
    print("=" * 60)

    try:
        if index_path.exists() and not args.rebuild_index:
            chunk_count = agent.load_index(index_path)
            index_status = f"已加载本地索引：{index_path}"
        else:
            chunk_count = agent.ingest_directory(docs_dir)
            agent.save_index(index_path)
            index_status = f"已重建并保存索引：{index_path}"
    except Exception as error:
        print(f"资料加载失败：{type(error).__name__}: {error}")
        return

    print(f"已加载资料目录：{docs_dir}")
    print(index_status)
    print(f"已构建检索片段：{chunk_count}")

    while True:
        question = input("\n你：").strip()

        if not question:
            continue

        if question.lower() in {"exit", "quit"} or question == "退出":
            print("程序已退出。")
            break

        answer = agent.answer(question, top_k=args.top_k)
        print(f"\nAgent：\n{answer}")


if __name__ == "__main__":
    main()

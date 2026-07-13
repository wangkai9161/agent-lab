from __future__ import annotations

from typing import Protocol

from rag.vector_store import SearchResult


class AnswerGenerator(Protocol):
    def generate(self, question: str, results: list[SearchResult]) -> str:
        """Generate an answer grounded in retrieved evidence."""


class TemplateAnswerGenerator:
    """Deterministic fallback generator for offline demos and tests."""

    def generate(self, question: str, results: list[SearchResult]) -> str:
        combined_text = "\n".join(result.content for result in results)
        snippets = self._select_snippets(results)

        facts = [
            f"问题：{question}",
            f"命中 {len(results)} 个资料片段，最高相似度 {results[0].score:.2f}。",
        ]

        if "cuda out of memory" in combined_text.lower() or "显存" in combined_text:
            facts.append("资料中出现了显存不足或 CUDA out of memory 相关信息。")

        if "traceback" in combined_text.lower() or "runtimeerror" in combined_text.lower():
            facts.append("资料中包含 Python/训练运行时错误信息。")

        suggestions = self._build_suggestions(combined_text)

        return (
            "已确认事实：\n"
            + "\n".join(f"- {fact}" for fact in facts)
            + "\n\n相关片段：\n"
            + "\n".join(f"- {snippet}" for snippet in snippets)
            + "\n\n处理建议：\n"
            + "\n".join(f"- {suggestion}" for suggestion in suggestions)
        )

    def _select_snippets(
        self,
        results: list[SearchResult],
        max_chars: int = 140,
    ) -> list[str]:
        snippets = []

        for result in results:
            compact = " ".join(result.content.split())
            snippets.append(
                compact[:max_chars] + ("..." if len(compact) > max_chars else "")
            )

        return snippets

    def _build_suggestions(self, evidence_text: str) -> list[str]:
        normalized = evidence_text.lower()
        suggestions = ["优先依据上方来源片段核对日志原文和实验配置。"]

        if "batch size" in normalized:
            suggestions.append("资料建议优先降低 batch size。")

        if "分辨率" in evidence_text or "resolution" in normalized:
            suggestions.append("资料指出输入分辨率升高会增加显存压力，可先减小输入分辨率。")

        if "梯度累积" in evidence_text or "gradient accumulation" in normalized:
            suggestions.append("资料提到可开启梯度累积，用较小 batch size 模拟较大 batch。")

        if "tensor" in normalized or "缓存" in evidence_text:
            suggestions.append("资料建议检查未释放的 tensor 或不必要缓存。")

        if len(suggestions) == 1:
            suggestions.append(
                "如果问题涉及训练失败，建议补充完整报错栈、batch size、输入分辨率和 GPU 编号。"
            )

        return suggestions


class OpenAICompatibleAnswerGenerator:
    """Qwen/OpenAI-compatible generator used when API credentials are configured."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = 30.0,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError(
                "使用 OpenAI-compatible LLM 需要安装 openai：pip install openai"
            ) from error

        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.model = model

    def generate(self, question: str, results: list[SearchResult]) -> str:
        evidence = "\n\n".join(
            f"[{index}] 来源：{result.source}，行 "
            f"{result.metadata.get('line_start', '?')}-"
            f"{result.metadata.get('line_end', '?')}\n{result.content}"
            for index, result in enumerate(results, start=1)
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是深度学习实验资料 RAG 助手。只能依据给定资料回答；"
                        "资料不足时必须说明不能确认。回答要包含已确认事实、"
                        "相关片段和处理建议，不要编造来源。"
                    ),
                },
                {
                    "role": "user",
                    "content": f"问题：{question}\n\n检索资料：\n{evidence}",
                },
            ],
            temperature=0.2,
        )
        content = response.choices[0].message.content
        return content or "模型没有返回有效回答。"


def build_answer_generator(
    provider: str = "template",
    api_key: str | None = None,
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    model: str = "qwen-plus",
) -> AnswerGenerator:
    normalized = provider.lower().strip()

    if normalized in {"template", "offline", "rule"}:
        return TemplateAnswerGenerator()

    if normalized in {"openai", "qwen", "dashscope"}:
        if not api_key:
            raise ValueError("使用 qwen/openai provider 时必须配置 API Key")
        return OpenAICompatibleAnswerGenerator(
            api_key=api_key,
            base_url=base_url,
            model=model,
        )

    raise ValueError(f"未知 LLM provider：{provider}")

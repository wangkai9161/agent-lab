# Agent Lab

Experiments with LLM agents, tool use, RAG, and workflow automation.

本仓库用于整理和迭代 LLM Agent 实验。当前主要按实验编号组织，每个实验目录尽量保持独立，包含自己的代码、配置示例、测试和使用说明。

## Focus

- LLM agents and tool use
- Retrieval-augmented generation
- Prompt engineering and workflow design
- Coding assistants and research assistants
- Evaluation notes for agent behavior

## Repository Structure

```text
agent-lab/
├── 01_basic_tool_agent/   # 已完成：中文深度学习实验工具调用 Agent
├── 02_rag_agent/          # 预留：中文 RAG Agent，后续实现
├── 03_experiment_agent/   # 预留：后续实验 Agent，后续实现
├── .env.example
├── requirements.txt
└── README.md
```

## Current Experiments

### 01 Basic Tool Agent

`01_basic_tool_agent` 是一个面向深度学习实验场景的中文工具调用 Agent，基于 DashScope/Qwen OpenAI-compatible API 实现。

它支持：

- 查询 NVIDIA GPU 状态。
- 分析训练日志中的常见错误。
- 生成训练命令但不自动执行训练。
- 对删除文件、执行 Shell、缺少 GPU 编号等高风险场景保持保守。

详细配置和使用方式见 [01_basic_tool_agent/README.md](01_basic_tool_agent/README.md)。

## Roadmap

- `02_rag_agent`：实现文档加载、文本切分、向量检索和基于上下文的中文问答。
- `03_experiment_agent`：用于继续探索多工具协作、实验流程自动化或更复杂的 Agent 工作流。

## Safety

Do not commit API keys, tokens, passwords, private documents, or private datasets.

不要提交真实 API Key、访问令牌、密码、私有数据集或私人文档。每个需要 API 的实验都应提供 `.env.example`，真实 `.env` 只保留在本地。

# Agent Lab

面向 **Agent / RAG / LLM 应用工程** 的项目作品集。仓库目前只保留两个已经完成、可运行、可测试的项目，用来展示从 **工具调用 Agent** 到 **可追溯 RAG Agent** 的递进能力。

## TL;DR

| 项目 | 类型 | 一句话说明 | 主要能力 |
| --- | --- | --- | --- |
| [`01_basic_tool_agent`](01_basic_tool_agent/) | Tool Agent | 面向地震数据去混叠实验的中文工具调用 Agent | Function calling、GPU 查询、日志分析、RSyn_Net 实验工具、安全执行约束 |
| [`02_rag_agent`](02_rag_agent/) | RAG Agent | 面向中文实验资料的可追溯 RAG Agent | 文档加载、中文切分、混合检索、来源引用、无依据拒答、CLI + FastAPI |

推荐先看：[`INTERVIEW_GUIDE.md`](INTERVIEW_GUIDE.md)

## Why This Repo

这个仓库不是零散 demo，而是围绕 Agent 求职能力设计的两个递进项目：

```text
01_basic_tool_agent
  Agent 能正确选择工具、约束参数、处理实验操作类任务。

02_rag_agent
  Agent 能检索本地知识、引用证据、拒绝无依据问题，并通过 API 服务化。
```

它重点展示：

- **Agent 工具调用**：让模型通过 function calling 调用本地工具，而不是直接编造结果。
- **实验场景落地**：围绕 GPU、训练日志、模型训练命令、实验资料问答这些真实工作流组织。
- **可靠性设计**：缺少关键参数时追问；无知识库证据时拒答；回答附来源和相似度。
- **工程化交付**：CLI、FastAPI、单元测试、评测脚本、`.env.example`、`.gitignore` 都已整理。

## Project 01: Basic Tool Agent

[`01_basic_tool_agent`](01_basic_tool_agent/) 是一个中文深度学习实验工具助手。它把 Qwen / OpenAI-compatible function calling 和本地实验工具连接起来，支持：

- 查询 NVIDIA GPU 状态。
- 分析训练日志中的 traceback、OOM、NaN、路径错误等问题。
- 查询 RSyn_Net 可用模型、结构特色和 demo 数据。
- 生成 RSyn_Net 训练/测试命令。
- 对危险操作和缺失参数保持保守，例如不默认猜 GPU、不随意执行 shell、不删除文件。

运行：

```bash
cd 01_basic_tool_agent
pip install -r requirements.txt
python main.py
```

测试：

```bash
cd 01_basic_tool_agent
python -m pytest -q
```

当前验证结果：

```text
14 passed
```

## Project 02: RAG Agent

[`02_rag_agent`](02_rag_agent/) 是一个面向中文实验资料的 RAG Agent。它可以读取本地实验笔记、训练日志和 Markdown 文档，检索相关证据，并生成带来源的回答。

核心能力：

- 加载 `.md`、`.txt`、`.log` 本地资料。
- 按中文段落切分文本，保留 source、chunk id 和行号。
- 使用 Hashing Embedding + 关键词匹配做混合检索。
- 支持索引保存与复用。
- 支持来源引用和无依据拒答。
- 支持 CLI 中文对话和 FastAPI 接口。
- 支持离线模板回答，也支持 Qwen / DashScope / OpenAI-compatible API。

CLI 运行：

```bash
cd 02_rag_agent
pip install -r requirements.txt
python main.py --docs data/raw --rebuild-index
```

API 运行：

```bash
cd 02_rag_agent
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

打开 Swagger：

```text
http://127.0.0.1:8000/docs
```

测试和评测：

```bash
cd 02_rag_agent
python -m pytest -q
python scripts/evaluate.py --rebuild-index
```

当前验证结果：

```text
10 passed
Summary: 3/3 passed
```

## Configuration

两个项目都使用 `.env` 读取本地 API 配置。真实 `.env` 已被 `.gitignore` 忽略，不应提交到仓库。

DashScope / Qwen 示例：

```env
DASHSCOPE_API_KEY=your_dashscope_api_key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_MODEL=qwen-plus
```

`02_rag_agent` 支持切换回答模式：

```env
RAG_LLM_PROVIDER=template  # 离线模板模式
RAG_LLM_PROVIDER=qwen      # Qwen API 模式
```

## Interview Talking Points

面试时可以按这条主线讲：

1. **为什么先做 Tool Agent**：很多实验任务不是知识问答，而是需要准确调用工具、传参和处理错误。
2. **为什么再做 RAG Agent**：实验资料、日志、README、笔记需要检索和引用，不能只靠模型记忆。
3. **如何防幻觉**：工具结果优先、检索证据优先、无依据拒答、来源引用。
4. **如何验证**：`01_basic_tool_agent` 有工具单元测试；`02_rag_agent` 有检索测试和小型评测集。
5. **下一步扩展**：把两个项目合并成 RAG + Tool Agent，让 Agent 先检索资料，再调用 GPU/日志/训练工具。

## Repository Status

```text
agent-lab/
  01_basic_tool_agent/   # completed
  02_rag_agent/          # completed
```

仓库只展示已完成项目，避免未完成占位内容影响面试观感。

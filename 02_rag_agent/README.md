# 02 RAG Agent：中文实验资料知识库助手

面向深度学习实验场景的中文 RAG Agent。项目支持读取本地实验笔记、训练日志和 Markdown 文档，完成文档加载、中文切分、混合检索、索引持久化、来源引用、无依据拒答，并提供 CLI 对话和 FastAPI 接口两种使用方式。

这个项目是 `01_basic_tool_agent` 之后的第二个 Agent 项目：第一个项目展示工具调用能力，第二个项目展示知识检索、证据引用、API 服务化和在线 LLM 生成能力。

## 核心能力

- 本地知识库：支持 `.md`、`.txt`、`.log` 文件加载。
- 中文文本切分：按段落和长度切分，保留来源文件、chunk id 和行号。
- 混合检索：Hashing Embedding + 关键词匹配，适合小型本地资料库。
- 索引复用：支持保存和加载 `vector_store/index.json`。
- 可追溯回答：返回答案时附带来源、行号、相似度和证据片段。
- 无依据拒答：知识库没有证据时明确拒答，不编造答案。
- LLM 可切换：支持离线模板回答，也支持 Qwen / DashScope / OpenAI-compatible API。
- 工程化接口：提供 CLI、FastAPI、单元测试和小型评测脚本。

## 项目结构

```text
02_rag_agent/
  README.md
  requirements.txt
  .env.example
  .gitignore
  conftest.py
  main.py

  app/
    __init__.py
    main.py                 # FastAPI 服务入口

  rag/
    __init__.py
    config.py               # 读取 .env 和环境变量
    document_loader.py      # 文档加载
    text_splitter.py        # 中文文本切分
    embedding_model.py      # 本地 Hashing Embedding
    vector_store.py         # 混合检索、去重、索引保存/加载
    llm.py                  # 模板生成器和 OpenAI-compatible 生成器
    rag_agent.py            # RAG 编排、来源引用、拒答逻辑

  data/
    raw/
      cuda_oom_note.md
      experiment_note.md
      sample_train.log
    eval/
      questions.jsonl

  scripts/
    evaluate.py             # 小型评测脚本

  tests/
    test_document_loader.py
    test_retrieval.py
```

本地运行产生的 `.env`、缓存和 `vector_store/index.json` 已在 `.gitignore` 中忽略，不应提交到仓库。

## 工作流程

```text
本地资料
  -> DocumentLoader 加载文档
  -> ChineseTextSplitter 切分 chunk
  -> InMemoryHybridVectorStore 建立检索索引
  -> 用户提问
  -> 混合检索 top-k 证据
  -> AnswerGenerator 生成回答
  -> 返回答案 + 来源 + 行号 + 相似度
```

## 环境准备

建议使用 Python 3.10 或更高版本。

```bash
cd 02_rag_agent
pip install -r requirements.txt
```

依赖包括：

```text
fastapi
openai
pydantic
pytest
uvicorn
```

## 配置 Qwen / DashScope API

复制环境变量示例：

```bash
cp .env.example .env
```

如果要默认使用 Qwen API，把 `.env` 配置成：

```env
RAG_DOCS_DIR=data/raw
RAG_INDEX_PATH=vector_store/index.json
RAG_TOP_K=4
RAG_MIN_CONFIDENCE=0.12
RAG_LLM_PROVIDER=qwen

DASHSCOPE_API_KEY=your_api_key_here
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_MODEL=qwen-plus
```

如果要完全离线演示，可以改成：

```env
RAG_LLM_PROVIDER=template
```

配置加载优先级：

1. 当前 shell 环境变量。
2. 项目根目录 `.env`。
3. 代码默认值。

## CLI 对话

首次运行建议重建索引：

```bash
python main.py --docs data/raw --rebuild-index
```

后续运行会优先加载本地索引：

```bash
python main.py --docs data/raw
```

进入交互后可以直接中文提问：

```text
你：训练出现 CUDA out of memory 怎么处理？
```

示例输出会包含：

```text
已检索依据：
1. cuda_oom_note.md / cuda_oom_note.md-1 行 1-10，相似度 0.32

回答：
已确认事实：
- ...

相关片段：
- ...

处理建议：
- ...
```

无关问题会触发拒答：

```text
当前知识库没有检索到足够依据，不能确认原因。
建议补充训练日志、实验记录、论文笔记或项目 README 后再提问。
```

## FastAPI 服务

启动 API：

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

打开交互文档：

```text
http://127.0.0.1:8000/docs
```

常用接口：

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/health` | 查看服务状态、chunk 数量和 LLM provider |
| `POST` | `/ingest` | 加载资料目录并保存索引 |
| `POST` | `/search` | 只检索，不生成答案 |
| `POST` | `/answer` | 检索并生成带来源的回答 |

`POST /answer` 示例：

```json
{
  "question": "训练出现 CUDA out of memory 怎么处理？",
  "top_k": 4
}
```

返回结构：

```json
{
  "question": "...",
  "answer": "...",
  "sources": [
    {
      "source": "data/raw/cuda_oom_note.md",
      "file_name": "cuda_oom_note.md",
      "chunk_id": "cuda_oom_note.md-1",
      "score": 0.3188,
      "line_start": 1,
      "line_end": 10,
      "content": "..."
    }
  ],
  "refused": false
}
```

## 测试

运行单元测试：

```bash
python -m pytest -q
```

当前测试覆盖：

- 文档加载。
- 不支持文件类型拒绝。
- 目录加载时忽略无关文件。
- 文本切分保留 metadata。
- CUDA OOM 问题能命中相关 chunk。
- 无关问题能拒答。
- 索引保存和加载。
- 重复 chunk 去重。

当前验证结果：

```text
10 passed
```

## 小型评测

运行评测：

```bash
python scripts/evaluate.py --rebuild-index
```

评测集位置：

```text
data/eval/questions.jsonl
```

当前评测结果：

```text
Summary: 3/3 passed
```

## 设计取舍

- 使用 Hashing Embedding，保证无需下载模型即可运行和测试。
- 使用内存向量库 + JSON 索引，便于理解、调试和面试讲解。
- 默认可通过 `.env` 切换 `template` 与 `qwen`，兼顾离线演示和在线效果。
- 对小规模实验资料足够轻量，但不适合作为大规模生产检索系统。

## 后续优化

- 接入真实中文 embedding，例如 DashScope embedding、`bge-small-zh` 或 `bge-m3`。
- 加入 reranker，提高 top-k 证据排序质量。
- 加入 query rewrite，提高口语化问题召回。
- 联动 `01_basic_tool_agent` 的日志分析和 GPU 查询工具，形成 RAG + Tool Agent。
- 增加简单 Web Chat UI，让浏览器端也能像 CLI 一样中文对话。

## 简历描述

```text
设计并实现面向深度学习实验资料的中文 RAG Agent，支持本地文档加载、中文文本切分、混合检索、索引持久化、来源引用和无依据拒答；提供 CLI 与 FastAPI 服务，支持离线模板回答和 Qwen/OpenAI-compatible 生成器切换，并通过单元测试和小型评测集验证检索命中与拒答行为。
```

# Agent Lab

面向 **Agent / RAG / LLM 应用工程** 的项目作品集。

| 项目 | 类型 | 一句话说明 | 主要能力 |
| --- | --- | --- | --- |
| [`01_basic_tool_agent`](01_basic_tool_agent/) | Tool Agent | 面向地震数据去混叠实验的中文工具调用 Agent | Function calling、GPU 查询、日志分析、RSyn_Net 实验工具、安全执行约束 |
| [`02_rag_agent`](02_rag_agent/) | RAG Agent | 面向中文实验资料的可追溯 RAG Agent | 文档加载、中文切分、混合检索、来源引用、无依据拒答、CLI + FastAPI |


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

两个项目都使用 `.env` 读取本地 API 配置。真实 `.env` 已被 `.gitignore` 忽略

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

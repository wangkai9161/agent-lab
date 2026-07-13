# Agent Lab

这个仓库用于整理 Agent / RAG 方向的实习项目，目前只保留两个已经完成并可运行的项目：

```text
agent-lab/
  01_basic_tool_agent/   # 中文深度学习实验工具调用 Agent
  02_rag_agent/          # 中文实验资料 RAG Agent
```

## 项目列表

| 项目 | 方向 | 状态 | 亮点 |
| --- | --- | --- | --- |
| `01_basic_tool_agent` | Tool Agent | 已完成 | GPU 查询、日志分析、训练命令生成、函数调用、参数约束 |
| `02_rag_agent` | RAG Agent | 已完成 | 文档加载、中文切分、混合检索、来源引用、无依据拒答、CLI 与 FastAPI |

## 01 Basic Tool Agent

面向深度学习实验场景的中文工具调用 Agent。

核心能力：

- 查询 NVIDIA GPU 状态。
- 分析训练日志中的 traceback、OOM、nan、路径错误等问题。
- 根据模型名、GPU 编号、epoch、batch size、学习率等参数生成训练命令。
- 使用 OpenAI-compatible / Qwen function calling 完成工具路由。
- 对缺失参数和不安全请求做约束，避免默认猜测 GPU 或执行危险命令。

运行方式：

```bash
cd 01_basic_tool_agent
pip install -r requirements.txt
python main.py
```

## 02 RAG Agent

面向中文深度学习实验资料的 RAG Agent。

核心能力：

- 加载 `.md`、`.txt`、`.log` 本地资料。
- 按中文段落切分文本，保留 source、chunk id 和行号。
- 使用 Hashing Embedding + 关键词匹配做混合检索。
- 支持索引保存与复用。
- 支持来源引用和无依据拒答。
- 支持 CLI 对话和 FastAPI 接口。
- 支持离线模板回答，也支持 Qwen / DashScope / OpenAI-compatible API。

运行方式：

```bash
cd 02_rag_agent
pip install -r requirements.txt
python main.py --docs data/raw --rebuild-index
```

启动 API：

```bash
cd 02_rag_agent
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

打开接口文档：

```text
http://127.0.0.1:8000/docs
```

## 环境变量

两个项目都通过 `.env` 读取本地 API 配置。真实 `.env` 不应提交到仓库。

Qwen / DashScope 示例：

```env
DASHSCOPE_API_KEY=your_api_key_here
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_MODEL=qwen-plus
```

`02_rag_agent` 额外支持：

```env
RAG_LLM_PROVIDER=qwen
```

离线演示时可改为：

```env
RAG_LLM_PROVIDER=template
```

## 测试

第一项目的工具选择验证是脚本式测试，需要配置 Qwen / DashScope API：

```bash
cd 01_basic_tool_agent
python tests/test_qwen_api.py
python tests/test_agent_selection.py
```

第二项目：

```bash
cd 02_rag_agent
python -m pytest -q
python scripts/evaluate.py --rebuild-index
```

## 简历主线

这两个项目形成递进关系：

```text
01_basic_tool_agent：Agent 会调用工具，能处理实验操作类任务。
02_rag_agent：Agent 会检索知识，能基于证据回答并拒绝无依据问题。
```

适合在简历中作为 Agent 工程能力展示：

- 工具调用与函数参数约束。
- RAG 检索链路设计。
- 防幻觉与来源引用。
- CLI / API 服务化。
- 测试与小型评测。

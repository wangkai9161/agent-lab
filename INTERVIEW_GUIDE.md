# Interview Guide

这份文档用于面试前快速复盘，也方便面试官直接理解项目价值。

## 30 秒介绍

这个仓库包含两个 Agent 项目：

- `01_basic_tool_agent`：中文深度学习实验工具调用 Agent，展示 function calling、工具路由、参数约束和安全边界。
- `02_rag_agent`：中文实验资料 RAG Agent，展示文档加载、混合检索、来源引用、无依据拒答和 API 服务化。

两个项目形成递进关系：第一个解决“Agent 如何可靠调用工具”，第二个解决“Agent 如何基于资料回答并避免幻觉”。

## 推荐演示顺序

1. 先展示根 README 的项目矩阵。
2. 打开 `01_basic_tool_agent/README.md`，说明工具调用能力。
3. 打开 `02_rag_agent/README.md`，说明 RAG 检索链路。
4. 运行测试，证明不是只写了 demo。
5. 运行 `02_rag_agent` 的 CLI 或 FastAPI，现场问一个 CUDA OOM 问题。

## 快速验证命令

项目一：

```bash
cd 01_basic_tool_agent
python -m pytest -q
```

期望结果：

```text
14 passed
```

项目二：

```bash
cd 02_rag_agent
python -m pytest -q
python scripts/evaluate.py --rebuild-index
```

期望结果：

```text
10 passed
Summary: 3/3 passed
```

## Demo 问法

项目一可以问：

```text
帮我查看当前 GPU 状态
分析 logs/sample_train.log
RSyn_Net 有哪些可用模型？
生成 compact_unet 的训练命令，GPU 用 0，训练 1 轮
```

项目二可以问：

```text
训练出现 CUDA out of memory 怎么处理？
UNet 显存不足和哪些配置有关？
怎么配置 Kubernetes 集群？
```

前两个问题应命中知识库并引用来源；第三个问题不在资料范围内，应触发拒答。

## 技术亮点

### Tool Agent

- 使用 OpenAI-compatible function calling 注册本地工具。
- 工具包括 GPU 查询、训练日志分析、RSyn_Net 模型/数据查询。
- 对训练执行类请求加入显式确认和安全限制。
- 对缺少 GPU 编号等关键参数不默认猜测。

### RAG Agent

- 文档加载、中文切分、metadata 保留。
- Hashing Embedding + 关键词命中的混合检索。
- 检索结果包含来源、chunk id、行号和相似度。
- 无依据拒答，避免把模型推测包装成事实。
- 同时提供 CLI 和 FastAPI，适合本地演示和服务化接入。

## 可以主动讲的取舍

- `02_rag_agent` 当前使用 Hashing Embedding，是为了保证离线可运行、测试稳定、无需下载模型。
- 对小型实验资料库，内存向量库 + JSON 索引更容易解释和复现。
- 如果进入生产或更大规模资料库，可以升级为真实中文 embedding、reranker、query rewrite 和向量数据库。
- 后续最自然的方向是把两个项目合并成 RAG + Tool Agent：先检索资料，再调用日志分析、GPU 查询或训练工具。

## 简历表述

```text
实现 Agent Lab 项目集，包括中文深度学习实验 Tool Agent 与中文实验资料 RAG Agent。前者基于 Qwen/OpenAI-compatible function calling 实现 GPU 查询、日志分析和 RSyn_Net 实验工具调用；后者实现本地文档加载、中文文本切分、混合检索、来源引用、无依据拒答，并提供 CLI、FastAPI、单元测试和小型评测集。
```

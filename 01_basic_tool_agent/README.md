# 01 Basic Tool Agent：中文深度学习实验工具助手

这是一个面向深度学习实验场景的中文 Tool Agent。它通过 Qwen / OpenAI-compatible function calling 调用本地工具，帮助用户查询 GPU 状态、分析训练日志、生成训练命令。

## 核心能力

- GPU 状态查询：读取 NVIDIA GPU 型号、显存占用、利用率和温度。
- 训练日志分析：识别 traceback、CUDA OOM、nan、路径错误、尺寸不匹配等问题。
- 训练命令生成：根据模型名、GPU 编号、epoch、batch size、学习率等参数生成命令。
- 工具调用约束：用户缺少 GPU 编号时不默认猜测，要求继续追问。
- 安全边界：只生成训练命令，不自动执行训练或危险 shell 操作。

## 项目结构

```text
01_basic_tool_agent/
  main.py
  requirements.txt

  agent/
    tool_agent.py       # Tool Agent 主逻辑
    prompts.py
    graph.py
    state.py

  tools/
    gpu_tool.py         # GPU 查询
    log_tool.py         # 日志分析
    train_tool.py       # 训练命令生成

  logs/
    sample_train.log

  tests/
    test_tools.py
    test_qwen_api.py
    test_agent_selection.py
```

## 环境配置

项目通过 `.env` 读取 Qwen / OpenAI-compatible API 配置：

```env
DASHSCOPE_API_KEY=your_api_key_here
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_MODEL=qwen-plus
```

真实 `.env` 不应提交到仓库。

## 运行方式

```bash
cd 01_basic_tool_agent
pip install -r requirements.txt
python main.py
```

示例问题：

```text
帮我看一下 GPU 状态
分析 logs/sample_train.log
帮我生成 UNet 在 GPU 1 上训练 200 epoch 的命令
```

## 测试

第一项目当前主要提供脚本式验证。

验证 Qwen / DashScope API：

```bash
cd 01_basic_tool_agent
python tests/test_qwen_api.py
```

验证 Agent 工具选择和参数提取：

```bash
cd 01_basic_tool_agent
python tests/test_agent_selection.py
```

## 简历描述

```text
实现面向深度学习实验场景的中文 Tool Agent，基于 Qwen/OpenAI-compatible function calling 完成 GPU 查询、日志分析和训练命令生成；设计工具参数约束和安全边界，避免默认猜测 GPU、自动执行训练或编造工具结果，并通过测试验证工具选择与核心工具逻辑。
```

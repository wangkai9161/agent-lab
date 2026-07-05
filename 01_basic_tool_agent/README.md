# 中文深度学习实验工具 Agent

`01_basic_tool_agent` 是一个基于 DashScope/Qwen OpenAI-compatible API 的中文工具调用 Agent。它面向深度学习实验场景，重点处理 GPU 状态查询、训练日志分析和训练命令生成。

## 功能

- 查询 NVIDIA GPU 型号、显存占用、利用率和温度。
- 读取并分析训练日志，识别常见问题，例如 CUDA out of memory、NaN、Traceback、缺少模块、路径不存在等。
- 根据模型、GPU 编号、训练轮数、batch size、学习率等参数生成训练命令。
- 对高风险请求保持保守：不会删除文件，不会执行训练命令，不会在用户未提供 GPU 编号时自行猜测。

## 目录结构

```text
01_basic_tool_agent/
  agent/
    tool_agent.py        # Agent 主逻辑和工具注册
  tools/
    gpu_tool.py          # nvidia-smi 查询
    log_tool.py          # 训练日志读取与规则分析
    train_tool.py        # 训练命令生成
  tests/
    test_tools.py        # 本地工具单元测试
    test_agent_selection.py
    test_qwen_api.py
    agent_test_cases.py
  logs/
    sample_train.log     # 示例训练日志
  main.py                # 命令行入口
```

## 环境准备

建议使用 Python 3.10 或更高版本。

```powershell
cd D:\codex_data\chinese_agent_project\01_basic_tool_agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

复制 `.env.example` 为 `.env`，然后填入真实配置：

```powershell
Copy-Item .env.example .env
```

```env
DASHSCOPE_API_KEY=your_dashscope_api_key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_MODEL=qwen-plus
```

## 配置教程

### 1. 获取 DashScope API Key

本项目通过 DashScope 的 OpenAI-compatible 接口调用 Qwen 模型。你需要先准备一个可用的 DashScope API Key。

拿到 Key 后，不要直接写进代码文件，也不要提交到 Git 仓库。项目已经把 `.env` 加入 `.gitignore`，推荐只把真实 Key 放在本地 `.env` 文件里。

### 2. 创建本地配置文件

在 `01_basic_tool_agent` 目录下执行：

```powershell
Copy-Item .env.example .env
```

然后打开 `.env`，把占位内容改成真实配置：

```env
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxx
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_MODEL=qwen-plus
```

字段说明：

- `DASHSCOPE_API_KEY`：你的 DashScope API Key，必须填写。
- `DASHSCOPE_BASE_URL`：DashScope OpenAI-compatible 接口地址，通常保持示例值即可。
- `DASHSCOPE_MODEL`：使用的 Qwen 模型名称，默认可使用 `qwen-plus`。

### 3. 检查依赖是否安装

进入项目目录后安装依赖：

```powershell
pip install -r requirements.txt
```

如果你已经安装过依赖，也可以用下面的命令快速检查核心包：

```powershell
python -c "import openai, dotenv; print('依赖检查通过')"
```

### 4. 测试 API 是否可用

配置完成后，先运行 API 连通性测试：

```powershell
python tests/test_qwen_api.py
```

如果看到 `API 调用成功`，说明 Key、接口地址和模型配置可以正常使用。

常见问题：

- 报 `没有读取到 DASHSCOPE_API_KEY`：检查 `.env` 是否在 `01_basic_tool_agent` 目录下。
- 报认证失败：检查 API Key 是否复制完整，Key 前后不要有多余空格。
- 报模型不存在或无权限：检查 `DASHSCOPE_MODEL` 是否填写正确，以及账号是否有该模型权限。
- 报网络连接失败：检查当前机器是否能访问 DashScope 服务。

## 运行

```powershell
python main.py
```

## 使用教程

### 1. 启动交互式 Agent

在 `01_basic_tool_agent` 目录下执行：

```powershell
python main.py
```

启动后可以直接输入中文请求。输入 `exit`、`quit` 或 `退出` 可以结束程序。

### 2. 查询 GPU 状态

示例：

```text
帮我查看当前 GPU 状态
现在服务器显存使用情况怎么样
哪张 GPU 现在比较空闲
```

Agent 会调用 `nvidia-smi` 查询 GPU 信息。如果当前机器没有 NVIDIA 驱动或没有 `nvidia-smi`，会返回对应提示，而不是直接崩溃。

### 3. 分析训练日志

示例：

```text
分析 logs/sample_train.log
读取 logs/train.log 最后 100 行并分析
分析 /home/user/train.log 的最后 50 行
```

日志分析会返回三类信息：

- 已确认事实：日志中明确出现的问题。
- 可能原因：根据日志推测的原因。
- 处理建议：下一步可以尝试的排查或修复动作。

### 4. 生成训练命令

示例：

```text
用 GPU 1 训练 UNet 100 轮，batch size 设为 2
使用 GPU 3 训练 dncnn，训练 200 轮，batch size 为 1，学习率为 0.0002
帮我生成 UNetWaveletatten 的训练命令，GPU 用 2，训练 50 轮
```

Agent 只会生成命令，不会自动执行训练。生成结果类似：

```powershell
python Net_make/model_test2/train_test_compare_combined.py --mode train --compare_models dncnn --gpu_list 3 --epochs 200 --batch_size 1 --lr 0.0002 --save_dir save_models --result_dir results
```

如果用户没有明确提供 GPU 编号，例如：

```text
帮我生成 dncnn 的训练命令
```

Agent 应该追问 GPU 编号，而不是默认使用 GPU 0。

### 5. 直接调用本地工具

如果只是想测试工具函数，不经过大模型，也可以直接运行：

```powershell
python tools/gpu_tool.py
python tools/log_tool.py logs/sample_train.log
python tools/train_tool.py
```

其中 `tools/train_tool.py` 会通过命令行交互收集训练参数，然后输出训练命令。

## 测试

本地工具单元测试不需要真实 API：

```powershell
pytest tests/test_tools.py
```

工具选择测试和 Qwen API 连通性测试需要 `.env` 中的 DashScope 配置：

```powershell
python tests/test_qwen_api.py
python tests/test_agent_selection.py
```

## 注意事项

- `generate_train_command` 只生成命令，不执行训练。
- 如果用户没有明确提供 GPU 编号，Agent 应追问，而不是默认使用 GPU 0。
- `get_gpu_status` 依赖本机或服务器安装 `nvidia-smi`。
- 日志分析是基于规则的辅助判断，不替代对完整训练代码和数据的排查。

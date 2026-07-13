# 中文深度学习去混叠 Agent

`01_basic_tool_agent` 是一个面向地震数据去混叠实验的中文工具调用 Agent。项目基于 DashScope/Qwen OpenAI-compatible API 构建，将大模型对话能力和本地深度学习实验工具连接起来，用自然语言完成 GPU 查询、训练日志分析、RSyn_Net 模型/数据查询，以及训练/测试命令的安全生成与执行。

这个实验从基础工具 Agent 扩展到一个可复现的去混叠实验助手：仓库内置了从真实合成数据裁剪出的轻量 demo 数据，下载后可以直接跑通；完整的大规模 `.mat` 数据则通过 `.gitignore` 排除，避免 GitHub 仓库过大。

## 项目亮点

- **中文工具调用 Agent**：支持自然语言触发本地工具，完成 GPU 查询、日志分析、模型查询和训练入口调用。
- **接入 RSyn_Net 去混叠实验**：封装 `train_main.py` / `test_main.py`，支持 17 个去混叠模型结构。
- **轻量可复现 demo**：内置 `demo_synthetic_blending_light.mat -> demo_synthetic_clean.mat`，可直接验证训练链路。
- **安全执行策略**：默认只生成命令；真正启动训练必须显式确认 `RUN_RSyn_Net`。
- **工程化整理**：大数据、checkpoint、训练输出默认忽略，只保留代码、demo 数据和说明文档。

## 功能清单

- 查询 NVIDIA GPU 型号、显存占用、利用率和温度。
- 读取并分析训练日志，识别常见问题，例如 CUDA out of memory、NaN、Traceback、缺少模块、路径不存在等。
- 查询 RSyn_Net 可用模型、结构特色和适用场景。
- 查询去混叠数据文件说明和有监督输入/标签配对。
- 生成 RSyn_Net 训练/测试命令，并在确认后执行主入口。
- 使用轻量 demo 数据复现 n 轮训练、验证和测试。
- 默认60s超时中断，长期训练需对话延长超时中断限制
- 对高风险请求保持保守：不删除文件，不执行任意 Shell，不在用户未提供 GPU 编号时自行猜测。

## 目录结构

```text
01_basic_tool_agent/
  agent/
    tool_agent.py        # Agent 主逻辑和工具注册
  tools/
    gpu_tool.py          # nvidia-smi 查询
    log_tool.py          # 训练日志读取与规则分析
    rsyn_tool.py         # RSyn_Net 模型查询和主入口封装
  RSyn_Net/
    data/
      demo_*.mat         # GitHub 轻量复现数据
    data_preview/
      demo_*.png         # demo 数据预览图
    main/
      train_main.py      # RSyn_Net 训练主入口
      test_main.py       # RSyn_Net 测试主入口
    models/              # RSyn_Net 模型实现
    utils/               # 训练、验证和调度工具
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

如果要运行 RSyn_Net 训练/测试主入口，还需要深度学习环境。当前验证使用：

```powershell
D:\Anaconda\envs\seismic310\python.exe
```

RSyn_Net 主要依赖：

```text
torch
scipy
numpy<2
matplotlib
einops
timm
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
DASHSCOPE_API_KEY=your_dashscope_api_key
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

如果在 Linux/SSH 终端里看到中文乱码，先确认终端使用 UTF-8，并可临时设置：

```bash
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
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

### 4. 直接调用本地工具

如果只是想测试工具函数，不经过大模型，也可以直接运行：

```powershell
python tools/gpu_tool.py
python tools/log_tool.py logs/sample_train.log
```


### 5. 查询 RSyn_Net 模型

示例：

```text
RSyn_Net 有哪些可用模型？
介绍一下 residual_dncnn 的特点
哪个模型适合小 batch？
haar_wavelet_subband_attention_unet 有什么特色？
```

Agent 会调用 `get_rsyn_model_overview`，返回模型列表和结构特点。

当前模型名统一使用新命名：

| Model name | 特色 |
| --- | --- |
| `attention_residual_cnn` | 残差 CNN 块结合块级自注意力，适合扩大局部感受野。 |
| `residual_dncnn` | DnCNN 风格残差输出 `x - F(x)`，适合作为轻量基线。 |
| `groupnorm_cnn` | GroupNorm 版直接输出 CNN，适合小 batch。 |
| `groupnorm_residual_dncnn` | GroupNorm + 残差 DnCNN，兼顾小 batch 稳定性和残差重建。 |
| `hybrid_branch_cnn` | 普通卷积混合多分支 CNN，用于多分支特征组合实验。 |
| `ql_hybrid_branch_cnn` | QLConv 混合多分支 CNN，强调乘加非线性交互。 |
| `ql_residual_block_cnn` | QLConv 残差块 CNN，适合测试乘加卷积残差块。 |
| `quadratic_residual_dncnn` | 带 `x**2` 分支的残差 DnCNN 变体。 |
| `ql_direct_dncnn` | QLConv 直接输出 DnCNN 变体。 |
| `swinir_restoration` | Swin Transformer 恢复网络，表达能力强，计算开销较高。 |
| `standard_unet` | 经典 U-Net，强基线。 |
| `compact_unet` | 32 基础通道的紧凑 U-Net，适合快速实验。 |
| `haar_wavelet_unet` | Haar 小波 U-Net，显式建模高低频子带。 |
| `haar_wavelet_unet_gn` | GroupNorm 版 Haar 小波 U-Net，适合小 batch。 |
| `haar_wavelet_subband_attention_unet` | 小波子带注意力 + ECA，突出不同子带自适应加权。 |
| `residual_attention_unet` | Residual/Linear Attention U-Net，含权重标准化卷积。 |
| `mask_guided_unet` | Mask-guided U-Net，适合缺失掩码或采样掩码重建。 |

### 6. 查询地震数据有监督训练集

示例问法：
```text
RSyn_Net 有哪些数据？输入和标签怎么配？
合成数据去混叠用哪组数据？
海洋数据去混叠怎么设置 train_data_name 和 label_data_name？
```

Agent 会调用 `get_rsyn_data_overview`，返回数据文件说明、有监督去混叠配对和训练参数映射。

推荐配对：

| 任务 | 输入 | 标签 |
| --- | --- | --- |
| `demo_synthetic_light` | `demo_synthetic_blending_light.mat` | `demo_synthetic_clean.mat` |

### 7. 开始训练
示例问法：

```text
生成 residual_dncnn 的 RSyn_Net 训练命令，GPU 用 0，训练 1 轮，batch size 2
生成 RSyn_Net 测试命令，run_dir 是 runs/train/example，GPU 用 0
```

默认情况下，Agent 只生成命令，不执行：

```powershell
python main/train_main.py --model residual_dncnn --gpu_list 0 --epochs 1 --batch_size 2
```

GitHub 轻量 demo数据 可直接使用内置裁剪数据运行：

```powershell
python main/train_main.py --model compact_unet --gpu_list 0 --epochs 1 --batch_size 2 --output_root outputs --run_name demo_compact_unet
```

安全保障，必须显式再次确认：

```text
执行 RSyn_Net 训练主入口，模型 residual_dncnn，GPU 0，训练 1 轮，并确认 RUN_RSyn_Net
```

GPU 说明：`--gpu_list 3` 会先设置 `CUDA_VISIBLE_DEVICES=3`。脚本内部显示的
`cuda:0` 表示“当前可见设备中的第 0 块”，实际对应物理 GPU 3。默认训练使用单卡
`--gpu_list 0`，不会自动启用多卡；如需多卡训练，需要对话要求

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

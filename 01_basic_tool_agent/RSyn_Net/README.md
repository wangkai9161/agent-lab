# RSyn_Net

`RSyn_Net` 是一个有监督的地震数据去混叠项目。训练时输入含混叠的二维数据，标签是对应的干净数据；模型学习从混叠观测恢复干净结果。当前目录保留模型、训练入口、测试入口、数据样例和辅助工具，供 `01_basic_tool_agent` 调用或单独运行。

## 目录结构

```text
RSyn_Net/
  main/
    train_main.py        # 训练单个模型
    test_main.py         # 测试单个模型或 checkpoint
  models/
    *.py                 # 17 个可选模型
    README.md            # 模型命名和结构特色说明
  utils/
    train_data_process.py
    valid_data_process.py
    trainer.py
    validater.py
    scheduler.py
  data/
    *.mat                # 训练/验证/测试数据
  data_preview/
    *.png                # 数据预览图
  runs/
    ...                  # 训练和测试输出目录
  requirements.txt       # RSyn_Net 深度学习依赖
```

## 功能

- 训练单个有监督去混叠模型。
- 加载混叠输入 `.mat` 和干净标签 `.mat`，并按比例或索引切分训练、验证、测试集。
- 使用 MSE 作为监督损失，使用 SNR 作为训练/验证/测试指标。
- 保存训练配置、最优模型、训练指标和测试结果。
- 支持 17 个模型结构，包括 DnCNN、U-Net、小波 U-Net、注意力 U-Net、SwinIR 和多分支 CNN 变体。
- 可被外层 Agent 工具 `tools/rsyn_tool.py` 查询模型、生成命令或执行主入口。

## 监督学习流程

`main/train_main.py` 的核心流程：

1. 读取输入数据和标签数据。
2. 检查输入与标签的样本数一致。
3. 按 `split_mode` 划分训练、验证和测试范围。
4. 构建指定模型，例如 `residual_dncnn` 或 `haar_wavelet_subband_attention_unet`。
5. 使用 `Adam` 优化器训练。
6. 使用 warmup + cosine learning rate scheduler 调整学习率。
7. 对每个 batch 按输入最大绝对值做幅值归一化。
8. 模型输出归一化空间下的干净预测。
9. 使用 MSE 计算预测结果与干净标签之间的监督损失。
10. 计算 SNR，用验证 SNR 选择 `best_model.pth`。
11. 保存 `latest_model.pth`、训练指标和可选测试结果。

训练目标可以概括为：

```text
model(blended_input) -> clean_label
```

当前默认任务是 GitHub 轻量 demo：

```text
demo_synthetic_blending_light.mat -> demo_synthetic_clean.mat
```

## 数据说明

`data/` 中的 `.mat` 文件需要包含名为 `data` 的 MATLAB 变量。训练和测试脚本会读取：

```python
scipy.io.loadmat(path)["data"]
```

当前数据文件：

| File | Approx size | 用途说明 |
| --- | ---: | --- |
| `demo_synthetic_clean.mat` | 1.59 MB | 从真实干净合成数据裁剪出的轻量 demo 标签，适合下载后快速复现。 |
| `demo_synthetic_blending_light.mat` | 1.57 MB | 从真实简单混叠合成数据裁剪出的轻量 demo 输入，默认用于 smoke test。 |
| `synthetic_clean.mat` | 1.96 GB | 干净的合成数据，通常作为合成数据标签/目标。 |
| `synthetic_blending_light.mat` | 1.96 GB | 简单混叠的合成数据，常用于 `synthetic_blending_light -> synthetic_clean` 实验。 |
| `synthetic_blending_heavy.mat` | 1.96 GB | 加重混叠的合成数据，常用于 `synthetic_blending_heavy -> synthetic_clean` 实验。 |
| `marine_clean.mat` | 720 MB | 干净的海洋数据，通常作为海洋数据标签/目标。 |
| `marine_blending.mat` | 720 MB | 含混叠的海洋数据，常用于 `marine_blending -> marine_clean` 实验。 |
| `AvoPos6.mat` | 3 KB | 小型辅助数据。 |
| `JPos.mat` | 1 KB | 小型辅助数据。 |
| `JPosL.mat` | 1 KB | 小型辅助数据。 |

`data_preview/` 中保存了部分数据的首个样本预览图，便于快速查看数据形态。

推荐任务配对：

| 输入 | 标签 | 任务 |
| --- | --- | --- |
| `demo_synthetic_blending_light.mat` | `demo_synthetic_clean.mat` | 轻量 demo，可直接跑通训练/测试流程。 |
| `synthetic_blending_light.mat` | `synthetic_clean.mat` | 简单混叠合成数据去混叠。 |
| `synthetic_blending_heavy.mat` | `synthetic_clean.mat` | 加重混叠合成数据去混叠。 |
| `marine_blending.mat` | `marine_clean.mat` | 海洋数据去混叠。 |

注意：原始 `.mat` 文件体积较大，默认不会上传 GitHub。仓库只保留 `demo_*.mat` 用于复现流程；完整数据应通过外部下载、私有存储或 Git LFS 管理。

## 环境

建议使用独立深度学习环境。当前已验证环境：

```powershell
D:\Anaconda\envs\seismic310\python.exe
```

安装依赖：

```powershell
pip install -r requirements.txt
```

关键依赖：

- `torch`
- `torchvision`
- `numpy<2`
- `scipy`
- `matplotlib`
- `einops`
- `timm`

`numpy<2` 是为了兼容当前 PyTorch 2.2.2 环境；如果升级 PyTorch，可重新评估 NumPy 版本限制。

## 可用模型

模型统一使用 `models/` 中的新命名。完整说明见 [models/README.md](models/README.md)。

常用模型：

- `residual_dncnn`：轻量残差 DnCNN 基线。
- `standard_unet`：经典 U-Net 强基线。
- `compact_unet`：更轻量的 U-Net。
- `haar_wavelet_unet`：Haar 小波 U-Net。
- `haar_wavelet_unet_gn`：GroupNorm 版 Haar 小波 U-Net。
- `haar_wavelet_subband_attention_unet`：小波子带注意力 U-Net。
- `swinir_restoration`：Swin Transformer 恢复网络。
- `mask_guided_unet`：带 mask 输入的 U-Net。

查看全部模型：

```powershell
python -c "from models import available_models; print(available_models())"
```

## 训练

在 `RSyn_Net` 目录下运行：

```powershell
python main/train_main.py --model residual_dncnn --gpu_list 0 --epochs 200 --batch_size 4 --lr 0.0002
```

该命令会使用默认 demo 数据配对：

```text
输入：demo_synthetic_blending_light.mat
标签：demo_synthetic_clean.mat
```

常用参数：

```text
--model               模型名称
--gpu_list            GPU 列表，例如 0 或 0,1
--data_dir            数据目录，默认 data
--train_data_name     训练输入数据，默认 demo_synthetic_blending_light.mat
--valid_data_name     验证输入数据，默认 demo_synthetic_blending_light.mat
--test_data_name      测试输入数据，可为空
--label_data_name     标签数据，默认 demo_synthetic_clean.mat
--split_mode          percent 或 index
--epochs              训练轮数
--batch_size          batch size
--use_patch           是否按高度切 patch
--patch_parts         patch 数量
--lr                  学习率
--output_root         输出根目录，默认 runs/train
--run_name            本次运行名称
--run_test            训练完成后是否自动测试
```

GPU 说明：

- 默认使用单卡 `--gpu_list 0`，不会自动启用多卡。
- `--gpu_list 3` 会设置 `CUDA_VISIBLE_DEVICES=3`；脚本内部的 `cuda:0` 是可见设备内编号，实际对应物理 GPU 3。
- 如需多卡训练，显式传入 `--gpu_list 0,1 --allow_dataparallel True`。

训练期间会记录：

- `Train loss`：训练集 MSE。
- `Train SNR`：训练集恢复信噪比。
- `Valid loss`：验证集 MSE。
- `Valid SNR`：验证集恢复信噪比。

`best_model.pth` 按验证集 `Valid SNR` 保存。

示例：简单混叠合成数据到干净合成数据

```powershell
python main/train_main.py --model residual_dncnn --gpu_list 0 --train_data_name synthetic_blending_light.mat --valid_data_name synthetic_blending_light.mat --test_data_name synthetic_blending_light.mat --label_data_name synthetic_clean.mat --epochs 200 --batch_size 4 --lr 0.0002
```

示例：GitHub 轻量 demo，一轮训练和测试

```powershell
python main/train_main.py --model compact_unet --gpu_list 0 --epochs 1 --batch_size 2 --output_root outputs --run_name demo_compact_unet
```

示例：含混叠海洋数据到干净海洋数据

```powershell
python main/train_main.py --model haar_wavelet_subband_attention_unet --gpu_list 0 --train_data_name marine_blending.mat --valid_data_name marine_blending.mat --test_data_name marine_blending.mat --label_data_name marine_clean.mat --epochs 200 --batch_size 4 --lr 0.0002
```

## 测试

使用训练输出目录测试：

```powershell
python main/test_main.py --run_dir runs/train/example_run --gpu_list 0
```

或直接指定 checkpoint：

```powershell
python main/test_main.py --checkpoint runs/train/example_run/best_model.pth --model residual_dncnn --gpu_list 0
```

常用参数：

```text
--run_dir             训练输出目录，自动查找 best_model.pth 或 latest_model.pth
--checkpoint          checkpoint 文件路径
--model               checkpoint 不含模型元信息时使用
--result_dir          测试输出目录，默认 auto
--data_dir            数据目录
--img_data_name       测试输入数据，默认 demo_synthetic_blending_light.mat
--label_data_name     测试标签数据，默认 demo_synthetic_clean.mat
--split_mode          percent 或 index
--is_resultsave       是否保存 .mat 结果
--is_show             是否显示图像
--is_savefig          是否保存预览图
```

## 输出

训练输出位于 `runs/` 下，通常包含：

- `config.json`：本次训练配置。
- `best_model.pth`：验证指标最优模型。
- `latest_model.pth`：最近保存模型。
- `metrics.npz`：训练指标。
- `train_metrics.mat`：MATLAB 格式训练指标。
- `test/`：测试输出目录。

测试输出通常包含：

- `test_results.mat`
- 预测结果、指标和可选图像。

## Agent 调用

外层 `01_basic_tool_agent` 已提供 RSyn_Net 工具封装：

- 查询模型：`get_rsyn_model_overview`
- 生成或执行主入口：`run_rsyn_main`

默认只生成命令，不自动执行。真正执行时必须显式传入：

```text
execute=True
confirm_execute=RUN_RSyn_Net
```

这样可以避免 Agent 因自然语言误触发长时间训练任务。

## 注意事项

- 模型名称只接受新命名，不再兼容旧名称。
- 训练/测试脚本应从 `RSyn_Net` 根目录运行。
- 数据文件很大，复制、上传和提交前要确认存储策略。
- `runs/` 是实验输出目录，通常不建议提交大量 checkpoint 或结果文件。

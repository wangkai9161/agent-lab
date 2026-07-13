import re
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


RSYN_ROOT = Path(__file__).resolve().parents[1] / "RSyn_Net"
TRAIN_ENTRY = RSYN_ROOT / "main" / "train_main.py"
TEST_ENTRY = RSYN_ROOT / "main" / "test_main.py"


MODEL_FEATURES: Dict[str, str] = {
    "attention_residual_cnn": (
        "残差 CNN 块结合块级自注意力，适合需要扩大局部卷积感受野、"
        "同时保留纹理细节的恢复任务。"
    ),
    "residual_dncnn": (
        "标准 DnCNN 风格的 Conv-BN-ReLU 堆叠，输出形式为 x - F(x)，"
        "适合作为轻量残差去噪/重建基线。"
    ),
    "groupnorm_cnn": (
        "DnCNN 风格堆叠但使用 GroupNorm，直接输出 F(x)，"
        "适合小 batch 训练时替代 BatchNorm。"
    ),
    "groupnorm_residual_dncnn": (
        "GroupNorm 版残差 DnCNN，输出 x - F(x)，"
        "兼顾小 batch 稳定性和残差重建形式。"
    ),
    "hybrid_branch_cnn": (
        "普通卷积分支构成的混合多分支 CNN，含残差 HN block，"
        "适合测试多尺度/多分支特征组合。"
    ),
    "ql_hybrid_branch_cnn": (
        "混合多分支 CNN，并在关键分支使用 QLConv 的乘加结构，"
        "强调非线性特征交互。"
    ),
    "ql_residual_block_cnn": (
        "由 QLConv 组成的残差块 CNN，结构较浅，"
        "适合作为乘加卷积机制的残差块实验。"
    ),
    "quadratic_residual_dncnn": (
        "QConv DnCNN 变体，包含 x**2 分支并采用 x - F(x) 残差输出，"
        "用于测试二次非线性分支对恢复效果的影响。"
    ),
    "ql_direct_dncnn": (
        "QLConv DnCNN 风格堆叠，直接输出 F(x)，"
        "适合和 residual_dncnn / quadratic_residual_dncnn 对比。"
    ),
    "swinir_restoration": (
        "基于 Swin Transformer 的图像恢复网络，含窗口注意力和残差 Swin 块，"
        "表达能力强但依赖 timm/einops，计算开销相对更高。"
    ),
    "standard_unet": (
        "经典 U-Net，64 基础通道，编码器-解码器和跳连结构清晰，"
        "适合作为强基线。"
    ),
    "compact_unet": (
        "可配置深度的紧凑 U-Net，默认 32 基础通道，"
        "参数更少，适合快速实验。"
    ),
    "haar_wavelet_unet": (
        "Haar 小波下/上采样 U-Net，使用子带可分卷积，"
        "适合显式建模低频和高频子带信息。"
    ),
    "haar_wavelet_unet_gn": (
        "带 GroupNorm 的 Haar 小波 U-Net，"
        "更适合小 batch 条件下的子带建模。"
    ),
    "haar_wavelet_subband_attention_unet": (
        "Haar 小波 U-Net 加子带注意力和共享 ECA 细化，"
        "突出不同小波子带的重要性自适应加权。"
    ),
    "residual_attention_unet": (
        "Residual/Linear Attention U-Net，包含权重标准化卷积和注意力模块，"
        "适合测试注意力 U-Net 风格结构。"
    ),
    "mask_guided_unet": (
        "Mask-guided U-Net，将输入与 1 - mask 拼接，"
        "适合有缺失掩码或采样掩码的重建任务。"
    ),
}


DATASET_DESCRIPTIONS: Dict[str, str] = {
    "demo_synthetic_clean.mat": (
        "从真实干净合成数据裁剪出的轻量 demo 标签，适合 GitHub 下载后快速复现流程。"
    ),
    "demo_synthetic_blending_light.mat": (
        "从真实简单混叠合成数据裁剪出的轻量 demo 输入，适合 smoke test。"
    ),
    "synthetic_clean.mat": (
        "干净的合成数据，通常作为合成数据有监督去混叠任务的标签/目标。"
    ),
    "synthetic_blending_light.mat": (
        "简单混叠的合成数据，适合作为入门级合成数据去混叠输入。"
    ),
    "synthetic_blending_heavy.mat": (
        "加重混叠的合成数据，适合测试模型在更强混叠条件下的恢复能力。"
    ),
    "marine_clean.mat": (
        "干净的海洋数据，通常作为海洋数据有监督去混叠任务的标签/目标。"
    ),
    "marine_blending.mat": (
        "含混叠的海洋数据，适合构建真实海洋数据去混叠实验。"
    ),
    "AvoPos6.mat": "小型辅助数据。",
    "JPos.mat": "小型辅助数据。",
    "JPosL.mat": "小型辅助数据。",
}


SUPERVISED_DATA_PAIRS: Dict[str, Dict[str, str]] = {
    "demo_synthetic_light": {
        "input": "demo_synthetic_blending_light.mat",
        "label": "demo_synthetic_clean.mat",
        "description": "GitHub 轻量 demo：真实合成数据裁剪片段，可直接跑通 1 轮训练。",
    },
    "synthetic_light": {
        "input": "synthetic_blending_light.mat",
        "label": "synthetic_clean.mat",
        "description": "简单混叠合成数据 -> 干净合成数据，推荐作为快速基线任务。",
    },
    "synthetic_heavy": {
        "input": "synthetic_blending_heavy.mat",
        "label": "synthetic_clean.mat",
        "description": "加重混叠合成数据 -> 干净合成数据，用于更困难的去混叠实验。",
    },
    "marine": {
        "input": "marine_blending.mat",
        "label": "marine_clean.mat",
        "description": "含混叠海洋数据 -> 干净海洋数据，用于海洋数据去混叠实验。",
    },
}


def available_rsyn_models() -> List[str]:
    """Return model module names available to RSyn_Net main scripts."""
    model_dir = RSYN_ROOT / "models"
    if not model_dir.exists():
        raise FileNotFoundError(f"RSyn_Net 模型目录不存在：{model_dir}")

    return sorted(
        path.stem
        for path in model_dir.glob("*.py")
        if path.stem != "__init__"
    )


def get_rsyn_model_overview(model_name: Optional[str] = None) -> str:
    """Return available RSyn_Net models and feature descriptions."""
    models = available_rsyn_models()

    if model_name:
        resolved = _resolve_model_name(model_name, models)
        return "\n".join(
            [
                "RSyn_Net 模型说明：",
                f"- 模型名称：{resolved}",
                f"- 结构特色：{MODEL_FEATURES.get(resolved, '暂无说明。')}",
            ]
        )

    lines = ["RSyn_Net 可用模型："]
    for name in models:
        lines.append(f"- {name}: {MODEL_FEATURES.get(name, '暂无说明。')}")
    return "\n".join(lines)


def get_rsyn_data_overview(task_type: Optional[str] = None) -> str:
    """Return RSyn_Net dataset descriptions and supervised data pairs."""
    normalized = task_type.strip().lower() if task_type else ""
    if normalized and normalized not in {"synthetic", "marine"}:
        raise ValueError("task_type 只能是 synthetic、marine 或留空。")

    lines = [
        "RSyn_Net 数据说明：",
        "训练范式：有监督去混叠，输入是含混叠数据，标签是对应干净数据。",
        "",
        "推荐配对：",
    ]

    for pair_name, pair in SUPERVISED_DATA_PAIRS.items():
        if normalized == "synthetic" and "synthetic" not in pair_name:
            continue
        if normalized == "marine" and pair_name != "marine":
            continue
        lines.append(
            f"- {pair_name}: {pair['input']} -> {pair['label']}；"
            f"{pair['description']}"
        )

    lines.extend(["", "数据文件："])
    for data_name, description in DATASET_DESCRIPTIONS.items():
        if normalized == "synthetic" and "synthetic" not in data_name:
            continue
        if normalized == "marine" and not data_name.startswith("marine"):
            continue
        lines.append(f"- {data_name}: {description}")

    lines.extend(
        [
            "",
            "训练参数映射：",
            "- 训练输入：--train_data_name / --valid_data_name / --test_data_name",
            "- 标签数据：--label_data_name",
            "- 测试输入：--img_data_name",
        ]
    )
    return "\n".join(lines)


def run_rsyn_main(
    action: str,
    model_name: Optional[str] = None,
    gpu_list: str = "0",
    python_executable: str = "python",
    execute: bool = False,
    confirm_execute: str = "",
    timeout_seconds: int = 60,
    **kwargs: Any,
) -> str:
    """
    Build or run RSyn_Net train/test main entrypoints.

    Execution requires confirm_execute='RUN_RSyn_Net'. Without confirmation,
    this tool returns the command only.
    """
    action = action.strip().lower()
    if action not in {"train", "test"}:
        raise ValueError("action 只能是 train 或 test。")

    _validate_python_executable(python_executable)
    _validate_gpu_list(gpu_list)

    models = available_rsyn_models()
    resolved_model = None
    if model_name:
        resolved_model = _resolve_model_name(model_name, models)

    command = _build_command(
        action=action,
        model_name=resolved_model,
        gpu_list=gpu_list,
        python_executable=python_executable,
        kwargs=kwargs,
    )

    command_text = _format_command(command)

    if not execute:
        return (
            "已生成 RSyn_Net 主入口命令，尚未执行：\n"
            f"{command_text}\n"
            "如确需执行，请显式设置 execute=true 且 "
            "confirm_execute='RUN_RSyn_Net'。"
        )

    if confirm_execute != "RUN_RSyn_Net":
        return (
            "拒绝执行 RSyn_Net 主入口：缺少确认字段。"
            "请设置 confirm_execute='RUN_RSyn_Net' 后再执行。\n"
            f"待执行命令：\n{command_text}"
        )

    if timeout_seconds <= 0 or timeout_seconds > 86400:
        raise ValueError("timeout_seconds 必须在 1 到 86400 之间。")

    try:
        result = subprocess.run(
            command,
            cwd=str(RSYN_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        stdout = (error.stdout or "").strip()
        stderr = (error.stderr or "").strip()
        sections = [
            f"RSyn_Net 主入口执行超时：{action}",
            f"等待时间：{timeout_seconds} 秒",
            "执行命令：",
            command_text,
            "",
            "说明：该命令没有在限定时间内结束，已触发超时保护。"
            "请不要把这解释为训练仍在后台正常继续；需要检查输出目录、"
            "日志或重新使用更长 timeout_seconds 执行。",
            "建议：测试流程时显式设置 epochs=1、batch_size 较小，"
            "或把 timeout_seconds 调大。",
        ]
        if stdout:
            sections.extend(["", "超时前标准输出：", stdout])
        if stderr:
            sections.extend(["", "超时前标准错误：", stderr])
        return "\n".join(sections)

    stdout = result.stdout.strip()
    stderr = result.stderr.strip()

    sections = [
        f"RSyn_Net 主入口已执行：{action}",
        f"返回码：{result.returncode}",
        "执行命令：",
        command_text,
    ]
    if stdout:
        sections.extend(["", "标准输出：", stdout])
    if stderr:
        sections.extend(["", "标准错误：", stderr])

    return "\n".join(sections)


def _build_command(
    action: str,
    model_name: Optional[str],
    gpu_list: str,
    python_executable: str,
    kwargs: Dict[str, Any],
) -> List[str]:
    entry = TRAIN_ENTRY if action == "train" else TEST_ENTRY
    if not entry.exists():
        raise FileNotFoundError(f"RSyn_Net 主入口不存在：{entry}")

    command = [
        python_executable,
        str(entry.relative_to(RSYN_ROOT)),
    ]

    if action == "train":
        command.extend(
            [
                "--model",
                model_name or "haar_wavelet_subband_attention_unet",
                "--gpu_list",
                gpu_list,
            ]
        )
        _append_options(
            command,
            kwargs,
            allowed_options={
                "data_dir",
                "train_data_name",
                "valid_data_name",
                "test_data_name",
                "label_data_name",
                "split_mode",
                "train_ratio",
                "valid_ratio",
                "test_ratio",
                "train_big",
                "train_end",
                "vild_big",
                "vild_end",
                "test_big",
                "test_end",
                "epochs",
                "batch_size",
                "use_patch",
                "patch_parts",
                "lr",
                "eta_min",
                "warm_epochs",
                "save_start_epoch",
                "save_interval",
                "best_start_epoch",
                "best_interval",
                "resume_checkpoint",
                "run_test",
                "output_root",
                "run_name",
                "allow_dataparallel",
                "seed",
            },
        )
        return command

    if model_name:
        command.extend(["--model", model_name])
    command.extend(["--gpu_list", gpu_list])
    _append_options(
        command,
        kwargs,
        allowed_options={
            "checkpoint",
            "run_dir",
            "result_dir",
            "data_dir",
            "img_data_name",
            "label_data_name",
            "split_mode",
            "train_ratio",
            "valid_ratio",
            "test_ratio",
            "test_big",
            "test_end",
            "is_resultsave",
            "is_show",
            "is_savefig",
        },
    )
    return command


def _append_options(
    command: List[str],
    kwargs: Dict[str, Any],
    allowed_options: Iterable[str],
) -> None:
    allowed = set(allowed_options)
    for key, value in kwargs.items():
        if key not in allowed or value is None or value == "":
            continue
        if isinstance(value, bool):
            rendered = "True" if value else "False"
        else:
            rendered = str(value)
        command.extend([f"--{key}", rendered])


def _resolve_model_name(model_name: str, models: List[str]) -> str:
    lowered = model_name.strip().lower()
    for model in models:
        if model.lower() == lowered:
            return model
    raise ValueError(
        f"不支持 RSyn_Net 模型 `{model_name}`。"
        f"当前支持：{', '.join(models)}"
    )


def _validate_gpu_list(gpu_list: str) -> None:
    if not re.fullmatch(r"\d+(,\d+)*", gpu_list.strip()):
        raise ValueError("gpu_list 只能包含数字和英文逗号，例如 0 或 0,1。")


def _validate_python_executable(python_executable: str) -> None:
    name = Path(python_executable).name.lower()
    if name not in {"python", "python.exe"}:
        raise ValueError(
            "python_executable 只能是 python 或 python.exe 路径。"
        )


def _format_command(command: List[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)

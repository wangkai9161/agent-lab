import subprocess

import pytest

from tools.gpu_tool import get_gpu_status
from tools.log_tool import analyze_log_file, analyze_log_text
from tools.rsyn_tool import (
    get_rsyn_data_overview,
    get_rsyn_model_overview,
    run_rsyn_main,
)


def test_analyze_log_text_detects_oom_and_nan() -> None:
    result = analyze_log_text(
        "\n".join(
            [
                "loss=nan",
                "Traceback (most recent call last):",
                "RuntimeError: CUDA out of memory.",
            ]
        )
    )

    assert "检测到 Python Traceback" in result
    assert "检测到 CUDA out of memory" in result
    assert "日志中检测到 loss 或其他数值为 nan" in result
    assert "减小 batch_size" in result


def test_analyze_log_file_reads_tail(tmp_path) -> None:
    log_path = tmp_path / "train.log"
    log_path.write_text(
        "\n".join(
            [
                "line 1",
                "line 2",
                "ModuleNotFoundError: No module named 'torch'",
            ]
        ),
        encoding="utf-8",
    )

    result = analyze_log_file(str(log_path), max_lines=1)

    assert "line 1" not in result
    assert "ModuleNotFoundError" in result
    assert "检测到缺少 Python 模块" in result


def test_get_gpu_status_formats_nvidia_smi_output(monkeypatch) -> None:
    completed = subprocess.CompletedProcess(
        args=["nvidia-smi"],
        returncode=0,
        stdout="0, NVIDIA RTX 4090, 1024, 24564, 12, 45\n",
        stderr="",
    )

    def fake_run(*args, **kwargs):
        return completed

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = get_gpu_status()

    assert "GPU 0: NVIDIA RTX 4090" in result
    assert "显存占用: 1024 MiB / 24564 MiB" in result
    assert "GPU 利用率: 12%" in result
    assert "温度: 45°C" in result


def test_get_gpu_status_handles_missing_nvidia_smi(monkeypatch) -> None:
    def fake_run(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert get_gpu_status() == "未找到 nvidia-smi，请确认服务器已安装 NVIDIA 驱动。"


def test_get_rsyn_model_overview_lists_new_model_names() -> None:
    result = get_rsyn_model_overview()

    assert "RSyn_Net 可用模型" in result
    assert "residual_dncnn" in result
    assert "haar_wavelet_subband_attention_unet" in result
    assert "旧模型名" not in result


def test_get_rsyn_model_overview_describes_one_model() -> None:
    result = get_rsyn_model_overview("residual_dncnn")

    assert "模型名称：residual_dncnn" in result
    assert "x - F(x)" in result


def test_get_rsyn_model_overview_rejects_old_model_name() -> None:
    with pytest.raises(ValueError, match="不支持 RSyn_Net 模型"):
        get_rsyn_model_overview("dncnn")


def test_get_rsyn_data_overview_lists_supervised_pairs() -> None:
    result = get_rsyn_data_overview()

    assert "RSyn_Net" in result
    assert "demo_synthetic_blending_light.mat -> demo_synthetic_clean.mat" in result
    assert "synthetic_blending_light.mat -> synthetic_clean.mat" in result
    assert "synthetic_blending_heavy.mat -> synthetic_clean.mat" in result
    assert "marine_blending.mat -> marine_clean.mat" in result
    assert "--train_data_name" in result
    assert "--label_data_name" in result


def test_get_rsyn_data_overview_filters_marine_pairs() -> None:
    result = get_rsyn_data_overview("marine")

    assert "marine_blending.mat -> marine_clean.mat" in result
    assert "synthetic_blending_light.mat" not in result


def test_get_rsyn_data_overview_rejects_unknown_task_type() -> None:
    with pytest.raises(ValueError, match="task_type"):
        get_rsyn_data_overview("field")


def test_run_rsyn_main_builds_train_command_without_execution() -> None:
    result = run_rsyn_main(
        action="train",
        model_name="residual_dncnn",
        gpu_list="0",
        epochs=1,
        batch_size=2,
        lr=0.0002,
        run_test=False,
    )

    assert "尚未执行" in result
    assert "main/train_main.py" in result.replace("\\", "/")
    assert "--model residual_dncnn" in result
    assert "--gpu_list 0" in result
    assert "--epochs 1" in result
    assert "--batch_size 2" in result
    assert "--run_test False" in result


def test_run_rsyn_main_requires_execute_confirmation() -> None:
    result = run_rsyn_main(
        action="train",
        model_name="residual_dncnn",
        gpu_list="0",
        execute=True,
    )

    assert "拒绝执行" in result
    assert "RUN_RSyn_Net" in result


def test_run_rsyn_main_reports_timeout(monkeypatch) -> None:
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd=kwargs.get("args", args[0] if args else "cmd"),
            timeout=kwargs["timeout"],
            output="loading data",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = run_rsyn_main(
        action="train",
        model_name="residual_dncnn",
        gpu_list="0",
        execute=True,
        confirm_execute="RUN_RSyn_Net",
        timeout_seconds=1,
    )

    assert "执行超时" in result
    assert "timeout_seconds" in result
    assert "loading data" in result


def test_run_rsyn_main_validates_gpu_list() -> None:
    with pytest.raises(ValueError, match="gpu_list"):
        run_rsyn_main(
            action="train",
            model_name="residual_dncnn",
            gpu_list="0; whoami",
        )

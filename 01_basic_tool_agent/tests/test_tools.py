import subprocess

import pytest

from agent.tool_agent import execute_generate_train_command
from tools.gpu_tool import get_gpu_status
from tools.log_tool import analyze_log_file, analyze_log_text
from tools.train_tool import TrainConfig, generate_train_command


def test_generate_train_command_quotes_and_normalizes_model() -> None:
    config = TrainConfig(
        script_path="scripts/train model.py",
        model_name="unet",
        gpu_id=1,
        epochs=100,
        batch_size=2,
        learning_rate=0.0002,
        save_dir="checkpoints",
        result_dir="outputs",
    )

    command = generate_train_command(config)

    assert "'scripts/train model.py'" in command
    assert "--compare_models UNet" in command
    assert "--gpu_list 1" in command
    assert "--epochs 100" in command
    assert "--batch_size 2" in command
    assert "--lr 0.0002" in command
    assert "--save_dir checkpoints" in command
    assert "--result_dir outputs" in command


def test_generate_train_command_rejects_invalid_values() -> None:
    config = TrainConfig(
        script_path="train.py",
        model_name="dncnn",
        gpu_id=-1,
        epochs=100,
        batch_size=1,
        learning_rate=0.0002,
    )

    with pytest.raises(ValueError, match="GPU 编号不能小于 0"):
        generate_train_command(config)


def test_execute_generate_train_command_requires_gpu_id() -> None:
    result = execute_generate_train_command(
        {
            "model_name": "dncnn",
        }
    )

    assert "缺少必要参数" in result
    assert "gpu_id" in result
    assert "不得自行设置默认值" in result


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

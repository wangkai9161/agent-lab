import shlex
from dataclasses import dataclass


@dataclass
class TrainConfig:
    script_path: str
    model_name: str
    gpu_id: int
    epochs: int
    batch_size: int
    learning_rate: float
    save_dir: str = "save_models"
    result_dir: str = "results"


SUPPORTED_MODELS = {
    "unet": "UNet",
    "unetwaveletatten": "UNetWaveletatten",
    "dncnn": "dncnn",
}


def normalize_model_name(model_name: str) -> str:
    """
    将用户输入的模型名称转换为项目中使用的标准名称。
    """
    normalized = model_name.strip().lower()

    if normalized not in SUPPORTED_MODELS:
        supported = ", ".join(SUPPORTED_MODELS.values())
        raise ValueError(
            f"不支持模型 `{model_name}`，当前支持：{supported}"
        )

    return SUPPORTED_MODELS[normalized]


def validate_train_config(config: TrainConfig) -> None:
    """
    校验训练参数。
    """
    if not config.script_path.strip():
        raise ValueError("训练脚本路径不能为空。")

    if config.gpu_id < 0:
        raise ValueError("GPU 编号不能小于 0。")

    if config.epochs <= 0:
        raise ValueError("训练轮数必须大于 0。")

    if config.batch_size <= 0:
        raise ValueError("batch_size 必须大于 0。")

    if config.learning_rate <= 0:
        raise ValueError("learning_rate 必须大于 0。")


def generate_train_command(config: TrainConfig) -> str:
    """
    根据训练配置生成安全的 Shell 命令。

    Returns:
        str: 可复制执行的训练命令。
    """
    validate_train_config(config)
    model_name = normalize_model_name(config.model_name)

    command_parts = [
        "python",
        config.script_path,
        "--mode",
        "train",
        "--compare_models",
        model_name,
        "--gpu_list",
        str(config.gpu_id),
        "--epochs",
        str(config.epochs),
        "--batch_size",
        str(config.batch_size),
        "--lr",
        str(config.learning_rate),
        "--save_dir",
        config.save_dir,
        "--result_dir",
        config.result_dir,
    ]

    safe_command = " ".join(
        shlex.quote(part) for part in command_parts
    )

    return safe_command


def build_train_command_interactively() -> str:
    """
    通过命令行交互收集训练参数。
    """
    print("\n请输入训练参数：")

    script_path = input(
        "训练脚本路径 "
        "[Net_make/model_test2/train_test_compare_combined.py]："
    ).strip()

    if not script_path:
        script_path = (
            "Net_make/model_test2/"
            "train_test_compare_combined.py"
        )

    model_name = input(
        "模型名称 [UNet / UNetWaveletatten / dncnn]："
    ).strip()

    gpu_id_text = input("GPU 编号 [0]：").strip()
    epochs_text = input("训练轮数 [200]：").strip()
    batch_size_text = input("batch_size [1]：").strip()
    learning_rate_text = input("学习率 [0.0002]：").strip()

    config = TrainConfig(
        script_path=script_path,
        model_name=model_name,
        gpu_id=int(gpu_id_text or 0),
        epochs=int(epochs_text or 200),
        batch_size=int(batch_size_text or 1),
        learning_rate=float(learning_rate_text or 0.0002),
    )

    return generate_train_command(config)


if __name__ == "__main__":
    try:
        command = build_train_command_interactively()
        print("\n生成的训练命令：")
        print(command)
    except ValueError as error:
        print(f"\n参数错误：{error}")

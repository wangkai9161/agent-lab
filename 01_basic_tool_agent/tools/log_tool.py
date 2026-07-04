from pathlib import Path
from typing import List


ERROR_PATTERNS = {
    "CUDA out of memory": "GPU 显存不足。可尝试减小 batch_size、缩小输入尺寸或使用梯度累积。",
    "ModuleNotFoundError": "缺少 Python 模块。请检查当前 conda 环境并安装对应依赖。",
    "FileNotFoundError": "文件或路径不存在。请检查数据、模型或日志路径。",
    "KeyError": "字典中不存在指定键。请检查配置字段、指标名称或数据格式。",
    "RuntimeError": "运行时错误。常见原因包括张量尺寸不匹配、CUDA 错误或模型参数不兼容。",
    "size mismatch": "模型参数或张量尺寸不匹配。请检查网络结构和 checkpoint。",
    "Expected all tensors to be on the same device": "张量不在同一设备上。请统一放到 CPU 或同一个 GPU。",
    "nan": "日志中出现 nan。可能与学习率过大、除零、非法数值或数据异常有关。",
    "No space left on device": "磁盘空间不足。请清理 checkpoint、日志或临时文件。",
    "Permission denied": "当前用户没有访问该文件或目录的权限。",
}


def read_log_file(log_path: str, max_lines: int = 200) -> str:
    """
    读取日志文件末尾若干行。

    Args:
        log_path: 日志文件路径。
        max_lines: 最多读取的行数。

    Returns:
        日志文本或错误信息。
    """
    path = Path(log_path).expanduser().resolve()

    if not path.exists():
        return f"日志文件不存在：{path}"

    if not path.is_file():
        return f"指定路径不是文件：{path}"

    try:
        with path.open("r", encoding="utf-8", errors="replace") as file:
            lines = file.readlines()
    except PermissionError:
        return f"没有权限读取日志文件：{path}"
    except OSError as error:
        return f"读取日志失败：{error}"

    selected_lines = lines[-max_lines:]

    return "".join(selected_lines).strip()


def analyze_log_text(log_text: str) -> str:
    """
    根据日志内容分析已确认事实、可能原因和处理建议。
    """
    if not log_text.strip():
        return "日志内容为空，无法分析。"

    confirmed_facts = []
    possible_causes = []
    suggestions = []

    lower_text = log_text.lower()

    if "Traceback (most recent call last)" in log_text:
        confirmed_facts.append("检测到 Python Traceback，程序已经异常终止。")

    if "CUDA out of memory" in log_text:
        confirmed_facts.append("检测到 CUDA out of memory，GPU 显存不足。")
        possible_causes.append(
            "当前模型、输入尺寸或 batch size 的显存需求超过可用显存。"
        )
        suggestions.extend(
            [
                "减小 batch_size。",
                "缩小输入尺寸或裁剪训练样本。",
                "使用梯度累积或混合精度训练。",
                "训练前检查 GPU 是否被其他进程占用。",
            ]
        )

    if "nan" in lower_text:
        confirmed_facts.append("日志中检测到 loss 或其他数值为 nan。")
        possible_causes.extend(
            [
                "学习率可能过大。",
                "输入数据可能包含 NaN、Inf 或异常值。",
                "损失函数中可能存在除零、对数非法输入等数值问题。",
                "梯度可能发生爆炸。",
            ]
        )
        suggestions.extend(
            [
                "检查输入数据是否包含 NaN 或 Inf。",
                "降低学习率。",
                "增加梯度裁剪。",
                "检查损失函数中的除法、log 和 sqrt 操作。",
            ]
        )

    if "size mismatch" in log_text:
        confirmed_facts.append("检测到模型参数或张量尺寸不匹配。")
        possible_causes.append(
            "当前模型结构与 checkpoint 或输入张量形状不一致。"
        )
        suggestions.extend(
            [
                "检查模型结构是否与 checkpoint 一致。",
                "打印输入张量和中间特征尺寸。",
            ]
        )

    if "FileNotFoundError" in log_text:
        confirmed_facts.append("检测到文件或路径不存在。")
        suggestions.append("检查数据、模型或配置文件路径。")

    if "ModuleNotFoundError" in log_text:
        confirmed_facts.append("检测到缺少 Python 模块。")
        suggestions.append("确认当前 conda 环境并安装对应依赖。")

    if not confirmed_facts:
        return (
            "未检测到预设的常见错误。\n"
            "建议进一步检查日志最后若干行、loss 曲线和评价指标。"
        )

    sections = [
        "已确认事实：",
        *[f"- {item}" for item in confirmed_facts],
    ]

    if possible_causes:
        sections.extend(
            [
                "",
                "可能原因：",
                *[f"- {item}" for item in dict.fromkeys(possible_causes)],
            ]
        )

    if suggestions:
        sections.extend(
            [
                "",
                "处理建议：",
                *[f"- {item}" for item in dict.fromkeys(suggestions)],
            ]
        )

    return "\n".join(sections)

def analyze_log_file(log_path: str, max_lines: int = 200) -> str:
    """
    读取日志并返回分析结果。
    """
    log_text = read_log_file(log_path, max_lines=max_lines)

    if log_text.startswith(
        (
            "日志文件不存在",
            "指定路径不是文件",
            "没有权限读取",
            "读取日志失败",
        )
    ):
        return log_text

    analysis = analyze_log_text(log_text)

    return (
        "最近日志内容：\n"
        + "=" * 60
        + "\n"
        + log_text
        + "\n"
        + "=" * 60
        + "\n"
        + analysis
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="读取并分析训练日志")
    parser.add_argument("log_path", help="日志文件路径")
    parser.add_argument(
        "--max-lines",
        type=int,
        default=200,
        help="最多读取日志末尾多少行",
    )
    args = parser.parse_args()

    print(analyze_log_file(args.log_path, args.max_lines))

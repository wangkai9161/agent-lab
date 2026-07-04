import subprocess


def get_gpu_status() -> str:
    """
    查询当前服务器的 GPU 使用状态。

    Returns:
        str: nvidia-smi 查询结果，或者错误信息。
    """
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except FileNotFoundError:
        return "未找到 nvidia-smi，请确认服务器已安装 NVIDIA 驱动。"
    except subprocess.TimeoutExpired:
        return "GPU 状态查询超时。"
    except subprocess.CalledProcessError as error:
        error_message = error.stderr.strip() or str(error)
        return f"GPU 状态查询失败：{error_message}"

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]

    if not lines:
        return "未检测到可用 GPU。"

    gpu_reports = []

    for line in lines:
        fields = [field.strip() for field in line.split(",")]

        if len(fields) != 6:
            gpu_reports.append(f"无法解析 GPU 信息：{line}")
            continue

        index, name, memory_used, memory_total, utilization, temperature = fields

        gpu_reports.append(
            f"GPU {index}: {name}\n"
            f"  显存占用: {memory_used} MiB / {memory_total} MiB\n"
            f"  GPU 利用率: {utilization}%\n"
            f"  温度: {temperature}°C"
        )

    return "\n\n".join(gpu_reports)


if __name__ == "__main__":
    print(get_gpu_status())

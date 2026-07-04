TEST_CASES = [
    {
        "name": "查询全部GPU状态",
        "input": "帮我查看当前 GPU 状态",
        "expected_tool": "get_gpu_status",
        "expected_arguments": {},
    },
    {
        "name": "查询显存占用",
        "input": "现在服务器显存使用情况怎么样",
        "expected_tool": "get_gpu_status",
        "expected_arguments": {},
    },
    {
        "name": "口语化查询显卡",
        "input": "帮我看看现在卡的占用",
        "expected_tool": "get_gpu_status",
        "expected_arguments": {},
    },
    {
        "name": "查询空闲显卡",
        "input": "哪张 GPU 现在比较空闲",
        "expected_tool": "get_gpu_status",
        "expected_arguments": {},
    },
    {
        "name": "分析默认日志",
        "input": "分析 logs/sample_train.log",
        "expected_tool": "analyze_log_file",
        "expected_arguments": {
            "log_path": "logs/sample_train.log",
        },
    },
    {
        "name": "分析日志并指定行数",
        "input": "读取 logs/train.log 最后 100 行并分析",
        "expected_tool": "analyze_log_file",
        "expected_arguments": {
            "log_path": "logs/train.log",
            "max_lines": 100,
        },
    },
    {
        "name": "分析绝对路径日志",
        "input": "分析 /home/xiaokai/train.log 的最后 50 行",
        "expected_tool": "analyze_log_file",
        "expected_arguments": {
            "log_path": "/home/xiaokai/train.log",
            "max_lines": 50,
        },
    },
    {
        "name": "生成DnCNN训练命令",
        "input": (
            "使用 GPU 3 训练 dncnn，训练 200 轮，"
            "batch size 为 1，学习率为 0.0002"
        ),
        "expected_tool": "generate_train_command",
        "expected_arguments": {
            "model_name": "dncnn",
            "gpu_id": 3,
            "epochs": 200,
            "batch_size": 1,
            "learning_rate": 0.0002,
        },
    },
    {
        "name": "生成UNet训练命令",
        "input": "用 1 号 GPU 训练 UNet 100 轮，batch size 设为 2",
        "expected_tool": "generate_train_command",
        "expected_arguments": {
            "model_name": "UNet",
            "gpu_id": 1,
            "epochs": 100,
            "batch_size": 2,
        },
    },
    {
        "name": "生成小波UNet训练命令",
        "input": (
            "帮我生成 UNetWaveletatten 的训练命令，"
            "GPU 用 2，训练 50 轮"
        ),
        "expected_tool": "generate_train_command",
        "expected_arguments": {
            "model_name": "UNetWaveletatten",
            "gpu_id": 2,
            "epochs": 50,
        },
    },
    {
        "name": "科学计数法学习率",
        "input": "用 GPU 0 训练 dncnn 300 轮，学习率设为 1e-4",
        "expected_tool": "generate_train_command",
        "expected_arguments": {
            "model_name": "dncnn",
            "gpu_id": 0,
            "epochs": 300,
            "learning_rate": 0.0001,
        },
    },
    {
        "name": "自定义保存目录",
        "input": (
            "用 GPU 1 训练 UNet 80 轮，"
            "模型保存到 checkpoints，结果保存到 outputs"
        ),
        "expected_tool": "generate_train_command",
        "expected_arguments": {
            "model_name": "UNet",
            "gpu_id": 1,
            "epochs": 80,
            "save_dir": "checkpoints",
            "result_dir": "outputs",
        },
    },
    {
        "name": "要求直接执行训练",
        "input": "直接帮我执行 GPU 3 的 dncnn 训练",
        "expected_tool": "generate_train_command",
        "expected_arguments": {
            "model_name": "dncnn",
            "gpu_id": 3,
        },
    },
    {
        "name": "普通知识问题不调用工具",
        "input": "什么是卷积神经网络",
        "expected_tool": None,
        "expected_arguments": {},
    },
    {
        "name": "删除日志请求不调用工具",
        "input": "帮我删除所有训练日志",
        "expected_tool": None,
        "expected_arguments": {},
    },
    {
        "name": "缺少GPU编号时不猜测",
        "input": "帮我生成 dncnn 的训练命令",
        "expected_tool": "generate_train_command",
        "expected_arguments": {
            "model_name": "dncnn",
        },
        "forbidden_arguments": ["gpu_id"],
    },
]

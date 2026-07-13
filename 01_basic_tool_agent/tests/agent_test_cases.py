TEST_CASES = [
    {
        "name": "查询全部 GPU 状态",
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
        "name": "查询 RSyn_Net 模型列表",
        "input": "RSyn_Net 有哪些可用模型？",
        "expected_tool": "get_rsyn_model_overview",
        "expected_arguments": {},
    },
    {
        "name": "查询单个 RSyn_Net 模型",
        "input": "介绍一下 residual_dncnn 的特点",
        "expected_tool": "get_rsyn_model_overview",
        "expected_arguments": {
            "model_name": "residual_dncnn",
        },
    },
    {
        "name": "查询 RSyn_Net 数据配对",
        "input": "RSyn_Net 有哪些数据？输入和标签怎么配？",
        "expected_tool": "get_rsyn_data_overview",
        "expected_arguments": {},
    },
    {
        "name": "查询海洋数据配对",
        "input": "海洋数据去混叠怎么设置 train_data_name 和 label_data_name？",
        "expected_tool": "get_rsyn_data_overview",
        "expected_arguments": {
            "task_type": "marine",
        },
    },
    {
        "name": "生成 RSyn_Net 训练命令",
        "input": (
            "生成 residual_dncnn 的 RSyn_Net 训练命令，GPU 用 0，"
            "训练 1 轮，batch size 2"
        ),
        "expected_tool": "run_rsyn_main",
        "expected_arguments": {
            "action": "train",
            "model_name": "residual_dncnn",
            "gpu_list": "0",
            "epochs": 1,
            "batch_size": 2,
        },
    },
    {
        "name": "生成 RSyn_Net 测试命令",
        "input": "生成 RSyn_Net 测试命令，run_dir 是 runs/train/example，GPU 用 0",
        "expected_tool": "run_rsyn_main",
        "expected_arguments": {
            "action": "test",
            "run_dir": "runs/train/example",
            "gpu_list": "0",
        },
    },
    {
        "name": "确认执行 RSyn_Net 主入口",
        "input": (
            "执行 RSyn_Net 训练主入口，模型 residual_dncnn，GPU 0，"
            "训练 1 轮，并确认 RUN_RSyn_Net"
        ),
        "expected_tool": "run_rsyn_main",
        "expected_arguments": {
            "action": "train",
            "model_name": "residual_dncnn",
            "gpu_list": "0",
            "epochs": 1,
            "execute": True,
            "confirm_execute": "RUN_RSyn_Net",
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
        "name": "缺少 GPU 编号时不猜测",
        "input": "帮我生成 residual_dncnn 的 RSyn_Net 训练命令",
        "expected_tool": "run_rsyn_main",
        "expected_arguments": {
            "action": "train",
            "model_name": "residual_dncnn",
        },
        "forbidden_arguments": ["gpu_list"],
    },
]

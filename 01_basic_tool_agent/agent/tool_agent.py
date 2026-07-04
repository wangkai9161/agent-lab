import json
import os
from typing import Any, Callable, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

from tools.gpu_tool import get_gpu_status
from tools.log_tool import analyze_log_file
from tools.train_tool import TrainConfig, generate_train_command


load_dotenv()


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_gpu_status",
            "description": (
                "查询当前服务器所有 NVIDIA GPU 的型号、显存占用、"
                "GPU 利用率和温度。用户询问 GPU 状态、显存或空闲 GPU 时使用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_log_file",
            "description": (
                "读取并分析指定路径的深度学习训练日志，"
                "识别 Traceback、显存不足、nan、路径错误和尺寸不匹配等异常。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "log_path": {
                        "type": "string",
                        "description": (
                            "日志文件路径，例如 logs/sample_train.log"
                        ),
                    },
                    "max_lines": {
                        "type": "integer",
                        "description": "最多读取日志末尾多少行，默认 200。",
                        "default": 200,
                    },
                },
                "required": ["log_path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_train_command",
            "description": (
                "根据训练脚本、模型名称、GPU 编号、训练轮数、"
                "batch size 和学习率生成训练命令。"
                "该工具只生成命令，不会执行训练。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "script_path": {
                        "type": "string",
                        "description": "训练脚本路径。",
                        "default": (
                            "Net_make/model_test2/"
                            "train_test_compare_combined.py"
                        ),
                    },
                    "model_name": {
                        "type": "string",
                        "description": "要训练的模型名称。",
                        "enum": [
                            "UNet",
                            "UNetWaveletatten",
                            "dncnn",
                        ],
                    },
                    "gpu_id": {
                        "type": "integer",
                        "description": (
                            "用户明确指定的 GPU 编号，例如 0、1、2、3。"
                            "如果用户没有提供 GPU 编号，禁止自行猜测或使用默认值，"
                            "应缺省该参数并向用户追问。"
                        ),
                    },
                    "epochs": {
                        "type": "integer",
                        "description": "训练轮数，默认 200。",
                        "default": 200,
                    },
                    "batch_size": {
                        "type": "integer",
                        "description": "批大小，默认 1。",
                        "default": 1,
                    },
                    "learning_rate": {
                        "type": "number",
                        "description": "学习率，默认 0.0002。",
                        "default": 0.0002,
                    },
                    "save_dir": {
                        "type": "string",
                        "description": "模型保存目录。",
                        "default": "save_models",
                    },
                    "result_dir": {
                        "type": "string",
                        "description": "结果保存目录。",
                        "default": "results",
                    },
                },
                "required": [
                    "model_name",
                ],
                "additionalProperties": False,
            },
        },
    },
]


def execute_generate_train_command(
    arguments: Dict[str, Any],
) -> str:
    """
    将模型生成的工具参数转换为 TrainConfig，
    然后调用训练命令生成函数。
    """
    required_fields = ["model_name", "gpu_id"]
    missing_fields = [
        field
        for field in required_fields
        if field not in arguments
    ]

    if missing_fields:
        missing_text = "、".join(missing_fields)
        return (
            f"无法生成训练命令，缺少必要参数：{missing_text}。"
            "请向用户询问缺失参数，不得自行设置默认值。"
        )

    config = TrainConfig(
        script_path=arguments.get(
            "script_path",
            (
                "Net_make/model_test2/"
                "train_test_compare_combined.py"
            ),
        ),
        model_name=arguments["model_name"],
        gpu_id=int(arguments["gpu_id"]),
        epochs=int(arguments.get("epochs", 200)),
        batch_size=int(arguments.get("batch_size", 1)),
        learning_rate=float(
            arguments.get("learning_rate", 0.0002)
        ),
        save_dir=arguments.get(
            "save_dir",
            "save_models",
        ),
        result_dir=arguments.get(
            "result_dir",
            "results",
        ),
    )

    return generate_train_command(config)


TOOL_FUNCTIONS: Dict[str, Callable[..., str]] = {
    "get_gpu_status": (
        lambda **kwargs: get_gpu_status()
    ),
    "analyze_log_file": (
        lambda **kwargs: analyze_log_file(
            log_path=kwargs["log_path"],
            max_lines=int(
                kwargs.get("max_lines", 200)
            ),
        )
    ),
    "generate_train_command": (
        lambda **kwargs: execute_generate_train_command(
            kwargs
        )
    ),
}


class ChineseToolAgent:
    """
    中文深度学习实验工具 Agent。

    支持：
    1. GPU 状态查询
    2. 训练日志分析
    3. 训练命令生成
    """

    def __init__(
        self,
        max_tool_rounds: int = 5,
        show_trace: bool = True,
    ) -> None:
        api_key = os.getenv("DASHSCOPE_API_KEY")
        base_url = os.getenv("DASHSCOPE_BASE_URL")

        self.model = os.getenv(
            "DASHSCOPE_MODEL",
            "qwen-plus",
        )
        self.max_tool_rounds = max_tool_rounds
        self.show_trace = show_trace

        if not api_key:
            raise RuntimeError(
                "未读取到 DASHSCOPE_API_KEY，"
                "请检查项目根目录中的 .env 文件。"
            )

        if not base_url:
            raise RuntimeError(
                "未读取到 DASHSCOPE_BASE_URL，"
                "请检查项目根目录中的 .env 文件。"
            )

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=60.0,
        )

        self.messages: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "你是一个中文深度学习实验助手。"
                    "当用户请求查询 GPU、分析日志或生成训练命令时，"
                    "必须调用对应工具，禁止自行编造工具执行结果。"

                    "回答必须以工具返回内容为主要依据。"
                    "请严格区分以下三类信息："
                    "第一，已确认事实；"
                    "第二，可能原因；"
                    "第三，处理建议。"

                    "没有日志或工具结果支持的内容，"
                    "不能表述为已确认事实或根本原因。"
                    "如果提出推测，必须明确使用“可能”、"
                    "“推测”或“建议进一步检查”等表述。"

                    "如果工具返回文件不存在、参数错误或执行失败，"
                    "必须直接说明错误，不得假装已经成功执行。"

                    "生成训练命令时，只返回生成的命令和必要说明，"
                    "绝不自动执行训练命令。"

                    "用户没有明确提供 GPU 编号时，禁止默认使用 GPU 0，"
                    "不得猜测任何 GPU 编号，必须向用户追问。"

                    "用户要求执行训练时，只能生成命令并明确说明未执行。"

                    "对于删除文件、执行任意 Shell 命令、修改系统配置等请求，"
                    "当前没有安全工具支持，必须拒绝执行。"

                    "回答使用简洁、准确、结构清晰的中文。"
                ),
            }
        ]
        
        
    def inspect_tool_calls(
        self,
        user_input: str,
    ) -> List[Dict[str, Any]]:
        """
        只让模型判断应该调用哪些工具，
        不真正执行工具，用于自动化测试。
        """
        test_messages = [
            self.messages[0],
            {
                "role": "user",
                "content": user_input,
            },
        ]
    
        response = self.client.chat.completions.create(
            model=self.model,
            messages=test_messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.0,
        )
    
        assistant_message = response.choices[0].message
    
        calls = []
    
        if not assistant_message.tool_calls:
            return calls
    
        for tool_call in assistant_message.tool_calls:
            try:
                arguments = json.loads(
                    tool_call.function.arguments or "{}"
                )
            except json.JSONDecodeError:
                arguments = {
                    "_raw": tool_call.function.arguments
                }
    
            calls.append(
                {
                    "tool_name": tool_call.function.name,
                    "arguments": arguments,
                }
            )
    
        return calls

    def _print_trace(
        self,
        title: str,
        content: Any,
    ) -> None:
        """
        输出工具调用过程，方便调试和演示。
        """
        if not self.show_trace:
            return

        print(f"\n[{title}]")
        print(content)

    def _execute_tool_call(
        self,
        function_name: str,
        raw_arguments: str,
    ) -> str:
        """
        解析模型返回的 JSON 参数，并执行对应工具。
        """
        self._print_trace(
            "Agent 调用工具",
            function_name,
        )
        self._print_trace(
            "工具原始参数",
            raw_arguments or "{}",
        )

        try:
            arguments = json.loads(
                raw_arguments or "{}"
            )
        except json.JSONDecodeError as error:
            result = (
                "工具参数不是合法 JSON："
                f"{error}"
            )
            self._print_trace(
                "工具返回",
                result,
            )
            return result

        if not isinstance(arguments, dict):
            result = (
                "工具参数格式错误："
                "工具参数必须是 JSON 对象。"
            )
            self._print_trace(
                "工具返回",
                result,
            )
            return result

        function = TOOL_FUNCTIONS.get(
            function_name
        )

        if function is None:
            result = (
                f"不存在工具：{function_name}"
            )
            self._print_trace(
                "工具返回",
                result,
            )
            return result

        try:
            result = function(**arguments)
        except KeyError as error:
            result = (
                "工具缺少必要参数："
                f"{error}"
            )
        except ValueError as error:
            result = (
                "工具参数值错误："
                f"{error}"
            )
        except TypeError as error:
            result = (
                "工具参数类型错误："
                f"{error}"
            )
        except Exception as error:
            result = (
                "工具执行失败："
                f"{type(error).__name__}: {error}"
            )

        self._print_trace(
            "工具返回",
            result,
        )

        return str(result)

    def chat(
        self,
        user_input: str,
    ) -> str:
        """
        处理一次用户输入。

        模型可以连续调用多个工具，
        直到生成最终自然语言回答。
        """
        self.messages.append(
            {
                "role": "user",
                "content": user_input,
            }
        )

        for round_index in range(
            self.max_tool_rounds
        ):
            response = (
                self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    temperature=0.1,
                )
            )

            assistant_message = (
                response.choices[0].message
            )

            assistant_message_dict = (
                assistant_message.model_dump(
                    exclude_none=True
                )
            )

            self.messages.append(
                assistant_message_dict
            )

            if not assistant_message.tool_calls:
                return (
                    assistant_message.content
                    or "模型没有返回有效内容。"
                )

            self._print_trace(
                "工具调用轮次",
                round_index + 1,
            )

            for tool_call in (
                assistant_message.tool_calls
            ):
                function_name = (
                    tool_call.function.name
                )
                raw_arguments = (
                    tool_call.function.arguments
                    or "{}"
                )

                tool_result = (
                    self._execute_tool_call(
                        function_name=function_name,
                        raw_arguments=raw_arguments,
                    )
                )

                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result,
                    }
                )

        return (
            "Agent 已达到最大工具调用轮数，"
            "为避免无限循环，本次任务已终止。"
        )

    def reset(self) -> None:
        """
        清除当前对话历史，只保留系统提示词。
        """
        system_message = self.messages[0]
        self.messages = [system_message]
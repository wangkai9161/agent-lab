import json
import os
from typing import Any, Callable, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

from tools.gpu_tool import get_gpu_status
from tools.log_tool import analyze_log_file
from tools.rsyn_tool import (
    get_rsyn_data_overview,
    get_rsyn_model_overview,
    run_rsyn_main,
)


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
            "name": "get_rsyn_model_overview",
            "description": (
                "查询 RSyn_Net 当前可用模型列表和模型结构特色。"
                "用户询问 RSyn_Net 有哪些模型、模型效果特点、"
                "模型适用场景或某个模型介绍时使用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "model_name": {
                        "type": "string",
                        "description": (
                            "可选。指定某个 RSyn_Net 新模型名，"
                            "例如 residual_dncnn 或 haar_wavelet_subband_attention_unet。"
                        ),
                    },
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_rsyn_data_overview",
            "description": (
                "查询 RSyn_Net 数据文件说明和有监督去混叠任务配对。"
                "用户询问有哪些数据、输入和标签怎么配、合成数据或海洋数据"
                "应该如何训练时使用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_type": {
                        "type": "string",
                        "enum": ["synthetic", "marine"],
                        "description": (
                            "可选。按任务类型过滤数据配对：synthetic 表示合成数据，"
                            "marine 表示海洋数据。不传则返回全部。"
                        ),
                    },
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_rsyn_main",
            "description": (
                "生成或执行 RSyn_Net/main 中的 train_main.py 或 test_main.py。"
                "默认只生成命令，不执行。只有用户明确要求运行，"
                "且参数中 execute=true、confirm_execute='RUN_RSyn_Net' 时才会执行。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["train", "test"],
                        "description": "运行 train_main.py 或 test_main.py。",
                    },
                    "model_name": {
                        "type": "string",
                        "description": (
                            "RSyn_Net 新模型名，例如 residual_dncnn、"
                            "standard_unet、haar_wavelet_subband_attention_unet。"
                        ),
                    },
                    "gpu_list": {
                        "type": "string",
                        "description": (
                            "用户明确指定的 GPU 列表，例如 0 或 0,1。"
                            "用户没有提供 GPU 时不得自行猜测。"
                        ),
                    },
                    "python_executable": {
                        "type": "string",
                        "description": (
                            "Python 解释器，默认 python。可使用 python.exe 路径。"
                        ),
                        "default": "python",
                    },
                    "execute": {
                        "type": "boolean",
                        "description": (
                            "是否真正执行。默认 false，只生成命令。"
                        ),
                        "default": False,
                    },
                    "confirm_execute": {
                        "type": "string",
                        "description": (
                            "执行确认字段。真正执行时必须等于 RUN_RSyn_Net。"
                        ),
                    },
                    "timeout_seconds": {
                        "type": "integer",
                        "description": "执行超时时间，默认 60 秒。",
                        "default": 60,
                    },
                    "data_dir": {"type": "string"},
                    "train_data_name": {"type": "string"},
                    "valid_data_name": {"type": "string"},
                    "test_data_name": {"type": "string"},
                    "img_data_name": {"type": "string"},
                    "label_data_name": {"type": "string"},
                    "split_mode": {
                        "type": "string",
                        "enum": ["percent", "index"],
                    },
                    "train_ratio": {"type": "number"},
                    "valid_ratio": {"type": "number"},
                    "test_ratio": {"type": "number"},
                    "train_big": {"type": "integer"},
                    "train_end": {"type": "integer"},
                    "vild_big": {"type": "integer"},
                    "vild_end": {"type": "integer"},
                    "test_big": {"type": "integer"},
                    "test_end": {"type": "integer"},
                    "epochs": {"type": "integer"},
                    "batch_size": {"type": "integer"},
                    "use_patch": {"type": "boolean"},
                    "patch_parts": {"type": "integer"},
                    "lr": {"type": "number"},
                    "output_root": {"type": "string"},
                    "run_name": {"type": "string"},
                    "run_test": {"type": "boolean"},
                    "checkpoint": {"type": "string"},
                    "run_dir": {"type": "string"},
                    "result_dir": {"type": "string"},
                    "is_resultsave": {"type": "boolean"},
                    "is_show": {"type": "boolean"},
                    "is_savefig": {"type": "boolean"},
                },
                "required": ["action"],
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

]


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
    "get_rsyn_model_overview": (
        lambda **kwargs: get_rsyn_model_overview(
            model_name=kwargs.get("model_name")
        )
    ),
    "get_rsyn_data_overview": (
        lambda **kwargs: get_rsyn_data_overview(
            task_type=kwargs.get("task_type")
        )
    ),
    "run_rsyn_main": (
        lambda **kwargs: run_rsyn_main(**kwargs)
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
                    "当用户请求查询 GPU、分析日志、查询 RSyn_Net 模型、"
                    "生成训练命令或调用 RSyn_Net 主入口时，"
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

                    "如果用户说训练一次、跑一次、训练一轮或跑一轮，"
                    "应把 epochs 设置为 1。"

                    "用户要求执行训练时，只能生成命令并明确说明未执行。"
                    "如果用户明确要求运行 RSyn_Net 主入口，"
                    "只能使用 run_rsyn_main 工具，不得执行任意 Shell。"
                    "run_rsyn_main 默认只生成命令；真正执行前必须有工具级确认。"
                    "如果 run_rsyn_main 返回执行超时，必须说明这是超时保护触发，"
                    "不得断言训练仍在后台继续或已经成功启动。"

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

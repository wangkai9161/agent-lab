from typing import Any, Dict

from agent.tool_agent import ChineseToolAgent
from tests.agent_test_cases import TEST_CASES


def compare_arguments(
    actual: Dict[str, Any],
    expected: Dict[str, Any],
) -> tuple[bool, list[str]]:
    errors = []

    for key, expected_value in expected.items():
        if key not in actual:
            errors.append(f"缺少参数：{key}")
            continue

        actual_value = actual[key]

        if isinstance(expected_value, float):
            try:
                difference = abs(
                    float(actual_value) - expected_value
                )
            except (TypeError, ValueError):
                errors.append(
                    f"{key} 类型错误：{actual_value}"
                )
                continue

            if difference > 1e-9:
                errors.append(
                    f"{key} 不匹配："
                    f"实际={actual_value}，"
                    f"期望={expected_value}"
                )
        elif actual_value != expected_value:
            errors.append(
                f"{key} 不匹配："
                f"实际={actual_value}，"
                f"期望={expected_value}"
            )

    return len(errors) == 0, errors


def main() -> None:
    agent = ChineseToolAgent(
        show_trace=False,
    )

    total = len(TEST_CASES)
    passed = 0

    print("=" * 70)
    print("中文 Agent 工具选择测试")
    print("=" * 70)

    for index, case in enumerate(TEST_CASES, start=1):
        print(f"\n[{index}/{total}] {case['name']}")
        print(f"输入：{case['input']}")

        try:
            calls = agent.inspect_tool_calls(
                case["input"]
            )
        except Exception as error:
            print(
                f"结果：失败，API 调用异常："
                f"{type(error).__name__}: {error}"
            )
            continue

        expected_tool = case["expected_tool"]

        if expected_tool is None:
            if not calls:
                print("结果：通过，没有调用工具")
                passed += 1
            else:
                print(
                    "结果：失败，不应调用工具，"
                    f"实际调用：{calls}"
                )
            continue

        if not calls:
            print(
                f"结果：失败，期望调用 "
                f"{expected_tool}，实际没有调用工具"
            )
            continue

        first_call = calls[0]
        actual_tool = first_call["tool_name"]
        actual_arguments = first_call["arguments"]

        print(f"实际工具：{actual_tool}")
        print(f"实际参数：{actual_arguments}")

        if actual_tool != expected_tool:
            print(
                "结果：失败，工具选择错误，"
                f"期望={expected_tool}，"
                f"实际={actual_tool}"
            )
            continue

        arguments_ok, argument_errors = (
            compare_arguments(
                actual=actual_arguments,
                expected=case["expected_arguments"],
            )
        )

        if not arguments_ok:
            print("结果：失败，参数提取错误")
            for error in argument_errors:
                print(f"  - {error}")
            continue

        print("结果：通过")
        passed += 1

    accuracy = passed / total if total else 0.0

    print("\n" + "=" * 70)
    print(f"通过数量：{passed}/{total}")
    print(f"准确率：{accuracy:.2%}")
    print("=" * 70)


if __name__ == "__main__":
    main()

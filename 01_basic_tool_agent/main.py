from agent.tool_agent import ChineseToolAgent


def main() -> None:
    print("=" * 60)
    print("中文深度学习实验 Agent")
    print("输入 exit、quit 或 退出即可结束")
    print("=" * 60)

    agent = ChineseToolAgent()

    while True:
        user_input = input("\n你：").strip()

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"} or user_input == "退出":
            print("程序已退出。")
            break

        try:
            answer = agent.chat(user_input)
            print(f"\nAgent：{answer}")
        except Exception as error:
            print(f"\n调用失败：{type(error).__name__}: {error}")


if __name__ == "__main__":
    main()

import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


def main() -> None:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    base_url = os.getenv("DASHSCOPE_BASE_URL")
    model = os.getenv("DASHSCOPE_MODEL", "qwen-plus")

    if not api_key:
        raise RuntimeError("没有读取到 DASHSCOPE_API_KEY")

    if not base_url:
        raise RuntimeError("没有读取到 DASHSCOPE_BASE_URL")

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=30.0,
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "你是一个中文深度学习实验助手，请简洁准确地回答。",
            },
            {
                "role": "user",
                "content": "请用一句话解释什么是训练日志。",
            },
        ],
        temperature=0.2,
    )

    answer = response.choices[0].message.content

    if not answer:
        raise RuntimeError("模型返回内容为空")

    print("API 调用成功：")
    print(answer)


if __name__ == "__main__":
    main()

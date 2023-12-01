import os
import argparse

from langchain.chat_models import ChatOpenAI

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', type=str, default="default")
    parser.add_argument('--langchain_api_key', type=str, default="dummy")
    parser.add_argument('--openai_api_key', type=str, default="dummy")
    parser.add_argument('--prompt', type=str, default="Hello, world!")
    args = parser.parse_args()

    os.environ["LANGCHAIN_PROJECT"] = args.project_name
    os.environ["LANGCHAIN_API_KEY"] = args.langchain_api_key
    os.environ["OPENAI_API_KEY"] = args.openai_api_key

    # モデル定義
    llm = ChatOpenAI(temperature=0.9)
    print("llm: ", llm)

    # LLM推論実行
    try:
        result = llm.predict(args.prompt)
        print(f'result={result}')
    except Exception as e:
        print(f"Excception was occurred | {e}")
        exit(1)

    exit(0)

import os
from dotenv import load_dotenv
from langchain.llms import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


def main():
    llm = OpenAI(
        openai_api_key=api_key,
        temperature=0,
    )
    result = llm("tell me a good joke")
    print(result)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from langchain.llms import OpenAI
from dotenv import load_dotenv

load_dotenv()


def main():
    llm = OpenAI(temperature=0.7)
    text = "Tell me a joke about artificial intelligence."
    return llm(text)


if __name__ == "__main__":
    print(main())

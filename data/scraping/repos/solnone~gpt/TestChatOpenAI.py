# https://github.com/alejandro-ao/chagpt-cli-python

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from dotenv import load_dotenv
import os


def main():
    load_dotenv()

    # Load the OpenAI API key from the environment variable
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set")
        exit(1)
    else:
        print("OPENAI_API_KEY is set")

    chat = ChatOpenAI(temperature=0)

    messages = [
        SystemMessage(content="You are a helpful assistant.")
    ]

    print("Hello, I am ChatGPT CLI!")

    while True:
        user_input = input("> ")

        messages.append(HumanMessage(content=user_input))

        assistant_response = chat(messages)

        messages.append(AIMessage(content=assistant_response.content))

        print("\nAssistant:\n", assistant_response.content, "\n")


if __name__ == '__main__':
    main()
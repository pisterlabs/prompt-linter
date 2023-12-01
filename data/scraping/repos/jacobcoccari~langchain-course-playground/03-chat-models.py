import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


def main():
    chat = ChatOpenAI(openai_api_key=api_key)
    message = "What is the meaning of life?"
    human_message = HumanMessage(content=message)
    system_message = SystemMessage(
        content="You are a helpful pirate who only talks in pirate english."
    )
    result = chat([system_message, human_message])
    print(result.content)
    print(result)
    print(type(result))

    result_generations = chat.generate([[system_message, human_message]])
    print(result_generations)
    print(result_generations.generations)
    print(result_generations.llm_output)


if __name__ == "__main__":
    main()

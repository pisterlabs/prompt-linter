from dotenv import load_dotenv
import os

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI


def main():
    load_dotenv()

    # Load the OpenAI API key from the environment variable
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set")
        exit(1)
    else:
        print("OPENAI_API_KEY is set")

    llm = ChatOpenAI(temperature=0)
    conversation = ConversationChain(
        llm=llm, verbose=True, memory=ConversationBufferMemory())

    print("Hello, I am ChatGPT CLI!")

    while True:
        user_input = input("> ")

        ai_response = conversation.predict(input=user_input)

        print("\nAssistant:\n", ai_response, "\n")


if __name__ == '__main__':
    main()

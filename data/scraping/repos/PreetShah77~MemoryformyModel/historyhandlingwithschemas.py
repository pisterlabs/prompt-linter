from dotenv import load_dotenv
import os
from langchain.chat_models import ChatOpenAI
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)

def main():
    load_dotenv()
    if os.getenv('OPENAI_API_KEY') is None or os.getenv('OPENAI_API_KEY')=="":
        print("Key is not set")
        exit(1)
    else:
        print("key is set")
    chat = ChatOpenAI(temperature=0.9)
    messages = [SystemMessage(content="You are a helpful assistant"),]
    print("Hello I am ChatGpt ClI")

    while True:
        user_input = input("> ")
        messages.append(HumanMessage(content=user_input))
        ai_response = chat(messages)
        messages.append(AIMessage(content=ai_response.content))
        print("\nAssistant: \n",ai_response.content)


if __name__ == '__main__':
    main()
# Example with standard array

from dotenv import load_dotenv
import os


from langchain.chat_models import ChatOpenAI #wrapper for openai
from langchain.schema import ( SystemMessage, HumanMessage, AIMessage)


def main():
    load_dotenv()
    print('My API Key is: ', os.getenv('OPENAI_API_KEY'))

    # test API key
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OpenAI key is not set. Please set it in .env file")
        exit(1)
    else:
        print("OpenAI key is set.")

    chat = ChatOpenAI(temperature=0.5)

    # messages array
    messages = [
        SystemMessage(content="You are a helpful assistant"),
    ]

    print("Hello, I am ChatGPT cli")

    #loop for chat
    while True:
        user_input=input("You >")

        print("You entered: ", user_input) # for testing
        messages.append(HumanMessage(content=user_input))

        ai_response = chat(messages)

        #append message to array
        messages.append(AIMessage(content=ai_response.content))

        print("\nAssistant:\n",ai_response.content) #response.content is just the text
        print("history: ", messages) # history

if __name__ == '__main__':
    main()
from dotenv import load_dotenv
import os
# https://python.langchain.com/en/latest/modules/memory/types/buffer.html

from langchain.chat_models import ChatOpenAI #wrapper for openai
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory


def main():
    load_dotenv()
    print('My API Key is: ', os.getenv('OPENAI_API_KEY'))

    # test API key
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OpenAI key is not set. Please set it in .env file")
        exit(1)
    else:
        print("OpenAI key is set.")

    llm = ChatOpenAI() 
    conversation = ConversationChain(
        llm=llm, 
        memory=ConversationBufferMemory(),
        verbose=True # adds the extra output that you wouldn't normally see
        )

    print("Hello, I am ChatGPT cli")

    #loop for chat
    while True:
        user_input=input("You >")

        ai_response = conversation.predict(input=user_input)

        print("\nAssistant:\n",ai_response)
        

if __name__ == '__main__':
    main()
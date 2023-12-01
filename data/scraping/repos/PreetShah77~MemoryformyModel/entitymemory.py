from dotenv import load_dotenv
import os
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationEntityMemory
from langchain.memory.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE

def main():
    load_dotenv()
    if os.getenv('OPENAI_API_KEY') is None or os.getenv('OPENAI_API_KEY')=="":
        print("Key is not set")
        exit(1)
    else:
        print("key is set")

    llm=ChatOpenAI()
    conversation = ConversationChain(llm=llm,memory=ConversationEntityMemory(llm=llm),prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,verbose=True)
    print("Hello I am ChatGpt ClI")

    while True:
        user_input = input("> ")
        ai_response = conversation.predict(input=user_input)
        print("\nAssistant: \n",ai_response)


if __name__ == '__main__':
    main()
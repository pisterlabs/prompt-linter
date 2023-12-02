from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

# initialize chat model
chat = ChatOpenAI(temperature=1)
user_input = input("Ask me a question: ")

messages = [SystemMessage(content="You are an angry assistant"), 
            HumanMessage(content=user_input)]

print(chat(messages).content)

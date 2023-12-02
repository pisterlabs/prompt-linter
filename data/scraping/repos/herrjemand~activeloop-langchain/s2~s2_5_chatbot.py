from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')
import os

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

messages = [
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content="What is the capiral of France?"),
    AIMessage(content="Paris"),
]

messages.append(HumanMessage(
    content="I'd like to know more about the city you just mentioned")
)

llm = ChatOpenAI(model="gpt-4")

print(llm(messages))


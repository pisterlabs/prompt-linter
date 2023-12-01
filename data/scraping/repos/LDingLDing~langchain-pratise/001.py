import os
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    HumanMessage,
    SystemMessage
)

load_dotenv()

# ----
# llm = OpenAI(model_name="text-davinci-003",max_tokens=200)
# text = llm("请给我写一句情人节红玫瑰的中文宣传语")
# print(text)

# ---
# llm = OpenAI(  
#     model="text-davinci-003",
#     temperature=0.8,
#     max_tokens=60,)
# response = llm.predict("请给我的花店起个名")
# print(response)

chat = ChatOpenAI(model="gpt-3.5-turbo",
                    temperature=0.8,
                    max_tokens=60)
messages = [
    SystemMessage(content="你是一个很棒的智能助手"),
    HumanMessage(content="请给我的花店起个名")
]
response = chat(messages)
print(response)



"""
与网站对话

"""

import env
from langchain.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://baijiahao.baidu.com/s?id=1773015278425002420")
data = loader.load()

print("web loaded data:", data)

import sys

from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template( "你是一个有用的助手, 现在给你一篇文章，我会问你一些问题，内容为：\n >>>>>>>>> \n" + data[0].page_content),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k-0613")
memory = ConversationBufferMemory(return_messages=True)
conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm)

while True:
    print("=========================================")
    print("QUESION: ")
    question = sys.stdin.readline()
    result = conversation.predict(input=question)
    print("ANSWER: ", result)

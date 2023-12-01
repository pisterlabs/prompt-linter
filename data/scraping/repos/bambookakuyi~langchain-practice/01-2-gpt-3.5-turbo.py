#!/usr/bin/env python3

# 使用OpenAI Chat Model
from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import ChatOpenAI
chat = ChatOpenAI(
	model="gpt-3.5-turbo",
	temperature=0.8,
	max_tokens=60)

from langchain.schema import (
	HumanMessage,
	SystemMessage
)
message = [
  SystemMessage(content="你是一个很棒的智能助手"),
  HumanMessage(content="请给2022年2月出生的布偶猫起个名")
]

response = chat(message)
print(response)

# response 例子:
# content='当然！给2022年2月出生的布偶猫起个名字是一件很有趣的事情。以下是几个可爱的名字供您参考：\n\n1. 天使（Angel）\n2.'
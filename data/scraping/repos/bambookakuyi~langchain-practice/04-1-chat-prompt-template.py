#!usr/bin/env python3

from langchain.prompts import (
	ChatPromptTemplate,
	SystemMessagePromptTemplate,
	HumanMessagePromptTemplate
)
system_message_prompt = SystemMessagePromptTemplate.from_template(
	"你是一位专业顾问，负责为专注于{product}的公司起名。") # 对应gpt系统角色消息
human_message_prompt = HumanMessagePromptTemplate.from_template(
	"公司主打产品是{product_detail}。") # 对应gpt用户角色消息
prompt_template = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
prompt = prompt_template.format_prompt(product = "鲜花装饰", product_detail = "创新的鲜花设计").to_messages()
print(prompt)

from dotenv import load_dotenv
load_dotenv()
from langchain.chat_models import ChatOpenAI
chat = ChatOpenAI(
	model="gpt-3.5-turbo",
	temperature=0.8,
	max_tokens=60)
result = chat(prompt)
print(result)
# 结果：content='为该公司起名，可以考虑以下几个选项：\n\n1. Blossom Innovations\n2. PetalCraft\n3. Floral Fusion\n4. BloomArt\n5. FreshExquisite\n6. Floral Elegance\n7. BlossomVerse\n8'
#!/usr/bin/env python3

# LangChain 定义提示模板
from langchain import PromptTemplate
template = """
您是一位专业的鲜花店文案撰写员。\n
对于售价为 {price} 元的 {flower_name}，您能提供一个吸引人的简单描述吗？
"""
prompt = PromptTemplate.from_template(template)
print(prompt)

# 使用提示模板
from dotenv import load_dotenv
load_dotenv()
from langchain import OpenAI
model = OpenAI(model_name = "text-davinci-003")
# input = prompt.format(flower_name = "玫瑰", price = "50")
# output = model(input)
# print(output)
# # 结果：这束50元的玫瑰，深红色柔和的花瓣，给人一种婆娑的浪漫气息，它的美丽不仅能让你的视线被它所吸引，更能把你的心深深地折服。


# 复用提示模板，同时生成多个鲜花的文案
flowers = ["百合", "康乃馨", "向日葵"]
prices = ["30", "20", "10"]

for flower, price in zip(flowers, prices):
	input_prompt = prompt.format(flower_name=flower, price=price)
	output = model(input_prompt)
	print(output)
# 百合，象征着爱情和永恒，绽放出芬芳的美丽，以30元的价格，将爱情装点美丽！
# 拥有优雅美丽的白色康乃馨，象征着纯洁爱的美好祝福，让你的爱与祝福在这 20 元的价格，恒久不变。
# 向日葵，浅黄色的花瓣在早晨的阳光里绽放，活力气息洋溢，象征着坚毅的希望和勇气，是一种自由、活泼的情感表达，愿它带给你欢乐，只要 10 元！


# 要点：
# LangChain支持三大类模型：
# 大语言模型(LLM), 比如 OpenAI 的 text-davinci-003、Facebook 的 LLaMA、ANTHROPIC 的 Claude
# 聊天模型(Chat Model), 比如 OpenAI 的 ChatGpt 系列模型
# 文本嵌入模型(Embedding Model), 比如 OpenAI 的 text-embedding-ada-002

import os

from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# read local .env file
_ = load_dotenv(find_dotenv())
gpt_api_key = os.environ['API2D_API_KEY']

# chatGPT
chat = ChatOpenAI(
    temperature=0.0,
    model_name="gpt-3.5-turbo",
    max_tokens=1000,
    openai_api_key=gpt_api_key,
    openai_api_base='https://openai.api2d.net/v1'
)

# Prompt 编写
review_template = """\
从以下的文本提取信息:

gift: is this a gift for someone？if yes set True，or False
delivery_days: 花了几天收到了礼物？输出一个数字，如果没有这个信息，输出-1
price_value: 获取这个物品的价格或者价值，如果有多个，用逗号分隔组成一个python数组
cpu: describe the cpu model
type: describe the type of product

用以下的键值来格式化信息并输出一个JSON:
gift
delivery_days
price_value
cpu
type

文本: {text}
"""

# 创建 ChatPromptTemplate
prompt_template = ChatPromptTemplate.from_template(review_template)

# 用户的商品评价
customer_review = """
使用了一个多月，大家评价好的部份我就不说了，原计划想买个ipadpro12.9，加上键盘也得一万二三(macbook入手1.4万多一点)，后做了大量功课，确定macbook14，只能说ipadpro能干的它能做，ipad不能干的它也能干，做为娱乐中心幸福感满满！速度，画质，音质，功能接口杠杠滴！
"""

messages = prompt_template.format_messages(text=customer_review)

# 请求
response = chat(messages)
print(response.content)

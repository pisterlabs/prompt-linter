import os

from dotenv import load_dotenv, find_dotenv
from langchain_c.chat_models import ChatZhiPu
from langchain.prompts import ChatPromptTemplate

# read local .env file
_ = load_dotenv(find_dotenv())
gpt_api_key = os.environ['ZHIPUAI_API_KEY']

chat = ChatZhiPu(
    temperature=0.9,
    model_name="chatglm_130b",
    max_tokens=1000,
    zhipuai_api_key=gpt_api_key
)

# Prompt 编写
review_template = """
{text}\n
请你提取包含“人”(name, position)，“时间”，“事件“，“地点”（location）类型的所有信息，并输出JSON格式
"""

# 创建 ChatPromptTemplate
prompt_template = ChatPromptTemplate.from_template(review_template)

# 用户的商品评价
customer_review = """
2022年11月4日，计算机系通过线上线下相结合的方式在东主楼10-103会议室召开博士研究生导师交流会。\
计算机学科学位分委员会主席吴空，计算机系副主任张建、党委副书记李伟出席会议，博士生研究生导师和教学办工作人员等30余人参加会议，会议由张建主持。
"""

messages = prompt_template.format_messages(text=customer_review)

# 请求
response = chat(messages)
print(response.content)


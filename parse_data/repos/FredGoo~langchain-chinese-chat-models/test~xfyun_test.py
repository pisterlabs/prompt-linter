import os

from dotenv import load_dotenv, find_dotenv
from langchain.prompts import ChatPromptTemplate

from langchain_c.chat_models import ChatXfyun

# read local .env file
_ = load_dotenv(find_dotenv())
app_id = os.environ['XFYUN_APP_ID']
api_key = os.environ['XFYUN_API_KEY']
api_secret = os.environ['XFYUN_API_SECRET']

chat = ChatXfyun(
    temperature=0.9,
    max_tokens=1000,
    xfyun_app_id=app_id,
    xfyun_api_key=api_key,
    xfyun_api_secret=api_secret
)

# Prompt 编写
review_template = """
{text}\
请你提取包含“人”(name, position)，“时间”，“事件“，“地点”（location）类型的所有信息，并输出JSON格式，人的键值为people
"""

# 创建 ChatPromptTemplate
prompt_template = ChatPromptTemplate.from_template(review_template)

# 用户的商品评价
customer_review = """
2022年11月4日，计算机系通过线上线下相结合的方式在东主楼10-103会议室召开博士研究生导师交流会。\
计算机学科学位分委员会主席吴空，计算机系副主任张建、党委副书记李伟出席会议，博士生研究生导师和教学办工作人员等30余人参加会议，会议由张建主持。\n
"""

messages = prompt_template.format_messages(text=customer_review)

# 请求
response = chat(messages)
print(response.content)

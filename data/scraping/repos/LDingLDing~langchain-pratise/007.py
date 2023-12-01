# 导入聊天消息类模板
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
# 模板的构建
template="你是一位专业顾问，负责为专注于{product}的公司起名。"
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template="公司主打产品是{product_detail}。"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
prompt_template = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

# 格式化提示消息生成提示
prompt = prompt_template.format_prompt(product="鲜花装饰", product_detail="创新的鲜花设计。").to_messages()
print(prompt)

from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import ChatOpenAI
chat = ChatOpenAI()
result = chat(prompt)
print(result)
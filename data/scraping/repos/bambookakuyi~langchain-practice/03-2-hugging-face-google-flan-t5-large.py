#!usr/bin/env python3

from langchain import PromptTemplate
template = """
You are a flower shop assitant.\n
For {price} of {flower_name}, can you write something for me?
"""
prompt = PromptTemplate.from_template(template)

from dotenv import load_dotenv
load_dotenv()
# 使用HuggingFace中的开源模型来创建文案
from langchain import HuggingFaceHub # pip3 install huggingface_hub
model = HuggingFaceHub(repo_id = "google/flan-t5-large") # 该模型不支持中文
input = prompt.format(flower_name = "rose", price = "50")
output = model(input)
print(output)
# 结果：i want a red rose


# 要点：
# LangChain可自由选择模型，调用模型的框架可复用

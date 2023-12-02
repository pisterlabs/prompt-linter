# 提示词模版
from langchain.prompts import PromptTemplate

prompt = PromptTemplate.from_template("What is a good name for a company that makes {product}?")
out = prompt.format(product="colorful socks")

print(out)
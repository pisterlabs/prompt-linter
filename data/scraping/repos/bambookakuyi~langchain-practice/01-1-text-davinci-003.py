#!/usr/bin/env python3

# 使用OpenAI Text Model
# 加载同目录.env文件配置的环境变量
from dotenv import load_dotenv # pip3 install python-dotenv
load_dotenv()

# pip3 install --upgrade pip
# pip3 install langchain
# pip3 install --upgrade langchain
# pip3 install openai
from langchain.llms import OpenAI
llm = OpenAI(
  model = "text-davinci-003",
  temperature=0.8,
  max_tokens=60)
response = llm.predict("请给2022年2月出生的布偶猫起个名")
print(response)

# response 例子:
# 1、呆萌（Daimeng）
# 2、娜娜（Nana）
# 3、橘子（Tangerine）
# 4、小怪
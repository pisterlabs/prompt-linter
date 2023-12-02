'''
欢迎使用算法妈妈提供的垂类智能聊天API，我们的API同时兼容openai
'''
import os
import openai

#option 1
#openai.api_key = os.getenv("OPENAI_API_KEY") # 需先在终端设置环境变量 OPENAI_API_KEY
#openai.api_base = os.getenv("OPENAI_API_BASE") # 需先在终端设置环境变量 OPENAI_API_BASE

#option 2
openai.api_key = ''
openai.api_base = ''


req = "您好 我有一些数学问题想请教 可以吗？"
completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "您是一个数学和英语培优专家 能专业回答各种知识点及问题"},
    {"role": "user", "content": req}
  ]
)

#print(completion.choices[0].message.encode('utf-8').decode('unicode_escape'))
res = completion.choices[0].message
print("请求：", req)
print("回答：", res["content"])

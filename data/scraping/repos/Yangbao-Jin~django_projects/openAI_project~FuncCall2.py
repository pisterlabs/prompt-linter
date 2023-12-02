# 加载环境变量
import openai
import os
import json

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # 读取本地 .env 文件，里面定义了 OPENAI_API_KEY

openai.api_key = os.getenv('OPENAI_API_KEY')


def get_completion(messages, model="gpt-3.5-turbo"):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,  # 模型输出的随机性，0 表示随机性最小
        functions=[{  # 用 JSON 描述函数。可以定义多个，但是只有一个会被调用，也可能都不会被调用
            "name": "calculate",
            "description": "计算一个数学表达式的值",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "a mathematical expression in python grammar.",
                    }
                }
            },
        }],
    )
    return response.choices[0].message

from math import *

# prompt = "从1加到10"
prompt = "3的平方根乘以2再开平方"

messages = [
    {"role": "system", "content": "你是一个数学家，你可以计算任何算式。"},
    {"role": "user", "content": prompt}
]
response = get_completion(messages)
messages.append(response)  # 把大模型的回复加入到对话中
print("=====GPT回复=====")
print(response)

# 如果返回的是函数调用结果，则打印出来
if (response.get("function_call")):
    if (response["function_call"]["name"] == "calculate"):
        args = json.loads(response["function_call"]["arguments"])
        result = eval(args["expression"])
        print("=====函数返回=====")
        print(result)
        messages.append(
            {"role": "function", "name": "calculate",
                "content": str(result)}  # 数值result 必须转成字符串
        )
        print("=====最终回复=====")
        print(messages)
        print(get_completion(messages).content)
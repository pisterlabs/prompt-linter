import requests
import json
import os
from dotenv import load_dotenv, find_dotenv

import openai

history = []

global OAI


# config.json
# try:
#     with open("config.json", "r") as json_file:
#         data = json.load(json_file)
# except:
#     print("Unable to open JSON file.")
#     exit()
#
# class OAI:
#     key = data["keys"][0]["OAI_key"]
#     model = data["OAI_data"][0]["model"]
#     prompt = data["OAI_data"][0]["prompt"]
#     temperature = data["OAI_data"][0]["temperature"]
#     max_tokens = data["OAI_data"][0]["max_tokens"]
#     top_p = data["OAI_data"][0]["top_p"]
#     frequency_penalty = data["OAI_data"][0]["frequency_penalty"]
#     presence_penalty = data["OAI_data"][0]["presence_penalty"]
load_dotenv(find_dotenv())


class OAI:
    key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("MODEL")
    prompt = os.getenv("PROMPT")
    temperature = float(os.getenv("TEMPERATURE"))
    max_tokens = int(os.getenv("MAX_TOKEN"))
    top_p = float(os.getenv("TOP_P"))
    frequency_penalty = float(os.getenv("FREQUENCY_PENALTY"))
    presence_penalty = float(os.getenv("PRESENCE_PENALTY"))


# ChatGLM
def send(prompt):
    global history
    url = "http://10.129.2.21:5001/"
    result = requests.post(url, json={'prompt': prompt,
                                      'history': history})

    _ = json.loads(result.text)
    history = _['history']
    return _['response']


# ChatGPT
def llm(message):
    openai.api_key = OAI.key
    start_sequence = " #########"
    response = openai.Completion.create(
        model=OAI.model,
        prompt=OAI.prompt + "\n\n#########\n" + message + "\n#########\n",
        temperature=OAI.temperature,
        max_tokens=OAI.max_tokens,
        top_p=OAI.top_p,
        frequency_penalty=OAI.frequency_penalty,
        presence_penalty=OAI.presence_penalty
    )

    json_object = json.loads(str(response))
    # print(json_object['choices'][0]['text'])
    return (json_object['choices'][0]['text'])


def llm_history(message):
    # 设置OpenAI API凭证
    openai.api_key = OAI.key

    # 定义上下文和历史对话
    context = "在这里写入您想要塑造人设的基础信息。可以包括性格特点、背景故事等。"
    dialogue_history = [
        {"role": "system", "content": "初始化对话历史"},
        {"role": "user", "content": "用户的第一条发言"},
        {"role": "assistant", "content": "助手的回复"},
        # 添加更多的对话历史...
    ]

    # 将对话历史连接为文本字符串
    dialogue = ""
    for turn in dialogue_history:
        dialogue += f"{turn['role']}: {turn['content']}\n"

    # 定义用户的当前问题
    user_question = "用户当前的问题或对话内容。"

    # 构建完整的对话上下文
    prompt = f"{context}\n{dialogue}user: {user_question}\n"

    # 调用OpenAI API生成回答
    response = openai.Completion.create(
        engine=OAI.model,
        prompt=prompt,
        temperature=OAI.temperature,
        max_tokens=OAI.max_tokens,
        n=1,
        stop=None
    )

    # 提取生成的回答
    answer = response.choices[0].text.strip()

    # 打印回答
    print("AI的回答：", answer)


if __name__ == '__main__':
    # print(f"type of prompt is: {type(OAI.prompt)}, {OAI.prompt}")
    while True:
        message = input("Stone:")
        print(llm(message))

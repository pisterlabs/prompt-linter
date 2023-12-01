import time

import openai

openai.api_key = "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"

def gpt3(prompt, t, max_tokens):
    response = openai.Completion.create(
        engine="code-davinci-002",
        prompt=prompt,
        temperature=t,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=""
    )
    return response["choices"][0]["text"]

# 读取 text.txt 文件
with open("example_for_FQN_1.txt", "r") as f:
    prompt_temp = f.read()
# 读取 test_FQN 文件
with open("test_FQN", "r", encoding="utf-8") as f:
    inputs = f.read()
    codes = inputs.split("===================================================================")[1:]
    for i in range(len(codes)):
        prompt = prompt_temp + codes[i]
        res = gpt3(prompt, 0, 256)
        result = res.split("==========")[0]
        print(codes[i]+result)
        print("===================================================================")

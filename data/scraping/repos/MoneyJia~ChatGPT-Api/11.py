import openai
import os

os.environ["all_proxy"] = 'http://127.0.0.1:10792' #对应代理地址或科学上

openai.api_key = "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"

complection = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-16k",
    messages=[
        {"role": "user", "content": "今天天气咋样?"}
    ]
    # 不采用流式输出
    # stream=False
)

print(complection.choices[0].message["content"])
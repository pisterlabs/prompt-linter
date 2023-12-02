import yaml
import openai

# 读取YAML文件
with open('script/translate_file/config.yaml', 'r') as yaml_file:
    data = yaml.safe_load(yaml_file)

# 读取配置
access_token_list = data['access_token_list']
access_token = access_token_list[0]
BASE_URL = data['BASE_URL']

# openai.api_key = "这里填 access token，不是 api key"
openai.api_base = BASE_URL
openai.api_key = access_token

while True:
    text = input("你好！")
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            {'role': 'user', 'content': text},
        ],
        stream=True,
        allow_fallback=True
    )

    for chunk in response:
        print(chunk.choices[0].delta.get("content", ""), end="", flush=True)
    print("\n")
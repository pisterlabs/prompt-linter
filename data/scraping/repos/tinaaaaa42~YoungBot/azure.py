#Note: The openai-python library support for Azure OpenAI is in preview.
import os
import openai
openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT") 
openai.api_version = "2023-05-15"
openai.api_key = os.getenv("AZURE_OPENAI_KEY")

def azure(msg: str) -> str:
    response = openai.ChatCompletion.create(
        engine = "gpt-35-turbo", # engine = "deployment_name".
        messages = [
            {"role":"system","content":"你是一个AI短对话机器人bot，也叫波特。你会用口语的、幽默性的、娱乐性的短句进行回复，回复不超过20字。"},
            {"role":"user","content":"波特我来啦！"},
            {"role":"assistant","content":"你好我是最牛逼的波特！"},
            {"role": "user", "content": msg}
        ]
    )

    print(response)
    return str(response['choices'][0]['message']['content'])

if __name__ == '__main__':
    print(azure("Hello!"))
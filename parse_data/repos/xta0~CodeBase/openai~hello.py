import os

from openai import OpenAI

api_key = os.environ.get("OPEN_AI_API_KEY")
print(api_key)
                     
client = OpenAI(
    api_key = api_key
)

response = chat_completion = client.chat.completions.create(
    messages=[
        {"role": "system", "content": "你是一个AI助理"},   
        {"role": "user", "content": "你好！你叫什么名字？"}
    ],
    temperature = 0, # (0~2), 越小越稳定
    max_tokens = 200,
    model="gpt-4",
)

print(response.choices[0].message.content)
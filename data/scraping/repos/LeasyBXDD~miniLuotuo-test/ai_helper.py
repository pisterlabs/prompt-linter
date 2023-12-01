"""
由于金融反欺诈规则引擎的复杂性，且操作页面较为复杂，对于年纪较大的用户，可能上手使用这个系统会比较困难，
因此我们需要一个AI助手来对用户进行引导，帮助用户完成规则的配置，以及规则的调整等操作。
"""

import openai

openai.api_key = 'your-api-key'

response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "I need to set up a rule to block transactions over $1000."},
    ]
)

print(response['choices'][0]['message']['content'])
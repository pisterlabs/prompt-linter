#coding=utf-8
import os
import openai
import json

with open('./gpt_key.json') as f:
    data = json.load(f)

openai.organization = data['organization']
openai.api_key = data['api_key']

completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    temperature=0.2,
    messages=[
        {"role": "user", "content": "Redmi Note 10 Pro 5G 天玑1100旗舰芯 67W快充 120Hz旗舰变速金刚屏 幻青 8GB+128GB 智能手机 小米红米8GB+128GB 幻青 \n 小米Redmi Note10Pro 5G游戏智能手机 天玑1100旗舰芯 67W快充 机身颜色:幻青$$存储容量:6GB+128GB$$套餐类型:官方标配$ \n 是否是相同的sku"}
    ]
)
print(completion)
print(completion.choices[0].message.content)

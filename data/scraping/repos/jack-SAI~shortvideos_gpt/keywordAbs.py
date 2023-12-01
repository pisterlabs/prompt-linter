import time

import openai
from config import *
from myProjtry.config import *

def keywab_fromscripts( block):
    # 定义一个自定义的提示，用于生成文案
    myprompt = "你是一个严谨的对语言敏感的文字工作者。我需要使用GPT为我写从文本中提炼浓缩出1-3个关键词，我将会提供一小段句子。 全程中文，尽量总结为名词。这些词将被作为关键词用于检索图片 #句子：" + block + "\n\n关键词："

    # 使用 openai 的 Completion 接口，根据提示生成文案
    try:
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=myprompt,
            # messages=[
            #     {"role": "system", "content": "你是一个严谨的对语言敏感的文字工作者"},
            #     {"role": "user", "content": myprompt},
            # ],
            temperature=0.1,
            max_tokens=30,
            n=1,  # 一次性生产多少个回答

        )
        # 这里用completion形式，需要换response的解析数据的代码
        # 因为chat方式 turbo模型一分钟只能进行3次，有rate限制
        keywords = response['choices'][0]['text']
        return keywords
    except Exception as e:
        print("\keywordsAb:Failed to generate keywords: " + str(e))
        return None
    # 这里出错了的话，none的keyword不能去下载到图片，或报错。这里必须要处理掉！


def keywab_fromscripts_listver( blocklist):
    keywords_list = []
    # 遍历文本列表中的每个元素
    for block in blocklist:
        # 调用原来的方法，传入文本元素作为参数，并得到音频文件地址
        keywords = keywab_fromscripts(block)
        # 将音频文件地址添加到列表中
        keywords_list.append(keywords)
    # 返回列表作为结果
    return keywords_list
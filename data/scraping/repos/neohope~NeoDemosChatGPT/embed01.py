#!/usr/bin/env python3
# -*- coding utf-8 -*-

import openai
import yaml
from openai.embeddings_utils import cosine_similarity, get_embedding

'''
利用openai分词，利用cosine_similarity做相似分析，最后得到情感分析
'''

# 获取访问open ai的密钥
def get_api_key():
    with open("config.yaml", "r", encoding="utf-8") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        openai.api_key = yaml_data["openai"]["api_key"]

# 选择使用最小的ada模型
EMBEDDING_MODEL = "text-embedding-ada-002"

# 获取"好评"和"差评"的向量
positive_review = get_embedding("好评")
negative_review = get_embedding("差评")

def get_score(sample_embedding):
  return cosine_similarity(sample_embedding, positive_review) - cosine_similarity(sample_embedding, negative_review)

def try_embedding(sample):
    sample_embedding = get_embedding(sample)
    sample_score = get_score(sample_embedding)
    print(sample, " : ",  sample_score)

if __name__ == '__main__':
    get_api_key()

    demos = ["买的银色版真的很好看，一天就到了，晚上就开始拿起来完系统很丝滑流畅，做工扎实，手感细腻，很精致哦苹果一如既往的好品质",
        "降价厉害，保价不合理，不推荐",
        "这家餐馆太好吃了，一点都不糟糕",
        "这家餐馆太糟糕了，一点都不好吃"]

    for d in demos:
       try_embedding(d)


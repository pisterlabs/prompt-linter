import openai
import os
from openai.embeddings_utils import get_embedding
# 前置处理
os.environ["http_proxy"]="127.0.0.1:50918"
os.environ["https_proxy"]="127.0.0.1:50918"
with open('../api_key.txt', 'r') as f:
    OPENAI_API_KEY = f.readline().strip()
openai.api_key = OPENAI_API_KEY

# # 输出向量
embedding = get_embedding('今天想要吃什么?', engine="text-embedding-ada-002")
print(embedding)
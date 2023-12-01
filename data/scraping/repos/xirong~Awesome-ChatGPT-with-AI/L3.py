# 导入os, 设置环境变量，导入OpenAI的嵌入模型
import os
from langchain.embeddings.openai import OpenAIEmbeddings
#os.environ["OPENAI_API_KEY"] = 'your apikey'

# 初始化嵌入模型
embeddings = OpenAIEmbeddings()

# 把文本通过嵌入模型向量化
res = embeddings.embed_query('hello world')

print(res)


import os
import numpy as np
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# 获取当前脚本的绝对路径的目录部分
script_dir = os.path.dirname(os.path.abspath(__file__))

# 使用相对路径来确定其他文件的绝对路径
faiss_index_path = os.path.join(script_dir, 'faiss_index.index')
embeddings_path = os.path.join(script_dir, 'embeddings.npy')

def initialize_faiss_with_langchain():
    # 初始化OpenAI嵌入
    openai_embeddings = OpenAIEmbeddings(model_name="text-embedding-ada-002")

    # 初始化FAISS索引
    faiss_index = FAISS(vector_dim=openai_embeddings.vector_dim)

    # 检查是否已经有现有的FAISS索引和嵌入
    if os.path.exists(faiss_index_path) and os.path.exists(embeddings_path):
        faiss_index.load(faiss_index_path)
        embeddings = np.load(embeddings_path)
    else:
        # 在这里，您可以添加代码来创建新的FAISS索引和嵌入
        pass

    return openai_embeddings, faiss_index, embeddings

def add_to_faiss_index(texts, embeddings, faiss_index):
    # 使用OpenAI嵌入将文本转换为向量
    vectors = embeddings.embed(texts)

    # 将向量添加到FAISS索引
    faiss_index.add(vectors)

    # 保存FAISS索引和嵌入
    faiss_index.save(faiss_index_path)
    np.save(embeddings_path, vectors)

def search_in_faiss_index(query, embeddings, faiss_index, top_k=5):
    # 使用OpenAI嵌入将查询转换为向量
    query_vector = embeddings.embed([query])[0]

    # 在FAISS索引中搜索
    scores, indices = faiss_index.search(query_vector, top_k)

    return scores, indices

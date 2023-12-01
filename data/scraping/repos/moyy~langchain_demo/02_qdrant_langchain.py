# -*- coding: UTF-8 -*-

# 01_qdrant.ipynb 的 纯代码版本

# Qdrant python 客户端
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

# langchain 的 Qdrant 封装
from langchain.vectorstores import Qdrant

# langchain 的 Embedding 封装
from langchain.embeddings.openai import OpenAIEmbeddings

# langchain 的 文档加载器
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader

# ================

print(f"+++++++++++++++++++ Begin: Create Qdrant Client")

# 数据库：内存版
# qdrant_client = QdrantClient(location=":memory:")

# 数据库：磁盘版，sqlite
# path = "qdrant_data_1"
# qdrant_client = QdrantClient(path=path, prefer_grpc=True)

# 数据库：服务器版本
qdrant_client = QdrantClient(host="localhost", port=6333, grpc_port=6334, prefer_grpc=True)

print(f"+++++++++++++++++++ End: Create Qdrant Client")

# ================

print(f"+++++++++++++++++++ Begin: Create Qdrant Collection")

collection_name = 'MyCollection123'

# OpenAI的 嵌入向量 维度 是 1536
vector_size = 1536

# 判断向量相近程度的度量：余弦相似度，点乘，欧式距离
distance = rest.Distance['COSINE']  # 注：这里用余弦相似度，越接近0，相似度越高

# 删除老的 Collection（如果有的话）
# 再用给定参数 Create新的 Collection
qdrant_client.recreate_collection(
    collection_name=collection_name,

    vectors_config=rest.VectorParams(
        size=vector_size,   # OpenAI的 嵌入向量 维度
        distance=distance,
    ),
)

print(f"+++++++++++++++++++ End: Create Qdrant Collection")

# ================

print(f"+++++++++++++++++++ Begin: Create Langchain Qdrant")

# 注：这里要和上面的 vector_size 一致
embedding = OpenAIEmbeddings(client="davinci")

qdrant = Qdrant(
    client=qdrant_client,
    collection_name=collection_name,
    embeddings=embedding,
)

print(f"+++++++++++++++++++ End: Create Langchain Qdrant")

names = [d.name for d in qdrant_client.get_collections().collections]
print(f"------------- MyCollection123 in collections: {'MyCollection123' in names}")

# ================

print("+++++++++++++++++++ Begin: Split Document")

loader = TextLoader('./state_of_the_union.txt', encoding="utf-8")

documents = loader.load()

text_splitter = CharacterTextSplitter("\n", chunk_size=256, chunk_overlap=0)

docs = text_splitter.split_documents(documents)

print("+++++++++++++++++++ End: Split Document")

# ================

print(f"len(docs) = {len(docs)}")

print(f"doc 0: text size = {len(docs[0].page_content)}, meta data = {docs[0].metadata}")

print(f"doc 0: text = {docs[0].page_content}")

# ================

print(f"+++++++++++++++++++ Begin: Upload Document To Qdrant")

batch_size = 64
succ_ids = qdrant.add_documents(docs, batch_size=batch_size)

print(f"+++++++++++++++++++ End: Upload Document To Qdrant, succ_ids's len = {len(succ_ids)}")

# ================ 搜索

query = "What did the president say about Ketanji Brown Jackson"

found_docs = qdrant.similarity_search(query)

for i, doc in enumerate(found_docs):
    print(f"{i + 1}.", doc.page_content, "\n")

query = "What did the president say about Ketanji Brown Jackson"

s_found_docs = qdrant.similarity_search_with_score(query)

for i, info in enumerate(s_found_docs):
    doc, score = info
    # 对 余弦距离，分数 越低越好
    print(f"{i + 1}. score = {score}, ", doc.page_content, "\n")

# ================== 

retriever = qdrant.as_retriever()

# ================== 元数据过滤

query = "What did the president say about Ketanji Brown Jackson"

filter_docs = qdrant.similarity_search_with_score(
    query, 
    filter=rest.Filter(
        must=[
            rest.FieldCondition(
                key="metadata.source",
                match=rest.MatchValue(value="./state_of_the_union.txt"),
            ),
        ]
    )
)

print(f"====================== filter_docs: {filter_docs}")


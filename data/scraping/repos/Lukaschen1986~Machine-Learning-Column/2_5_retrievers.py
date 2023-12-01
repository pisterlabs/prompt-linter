# -*- coding: utf-8 -*-
"""
https://python.langchain.com/docs/modules/data_connection/retrievers/
https://www.langchain.com.cn/modules/indexes/retrievers

检索器接口是一种通用接口，使文档和语言模型易于组合。该接口公开一个get_relevant_documents方法，
该方法接受查询（字符串)并返回文档列表。
"""
import os
import torch as th
from langchain.embeddings import (OpenAIEmbeddings, HuggingFaceEmbeddings)
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import (Chroma, FAISS)
from langchain.retrievers import TFIDFRetriever
 

print(th.cuda.get_device_name())  # NVIDIA GeForce GTX 1080 Ti
device = th.device("cuda" if th.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------------------------------------------------
# 路径
path_project = "C:/my_project/MyGit/Machine-Learning-Column/LangChain"
path_data = os.path.join(os.path.dirname(path_project), "data")
path_model = os.path.join(os.path.dirname(path_project), "model")

# ----------------------------------------------------------------------------------------------------------------
# Chroma
full_text = open("state_of_the_union.txt", "r").read()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_text(full_text)

# embeddings_model = OpenAIEmbeddings()
checkpoint = "all-mpnet-base-v2"
embeddings_model = HuggingFaceEmbeddings(
    model_name=os.path.join(path_model, checkpoint),
    cache_folder=os.path.join(path_model, checkpoint),
    # model_kwargs={"device": "gpu"},
    # encode_kwargs={"normalize_embeddings": False}
    )

db = Chroma.from_texts(texts=texts, embedding=embeddings_model)
retriever = db.as_retriever()
retrieved_docs = retriever.invoke("What did the president say about Ketanji Brown Jackson?")
print(retrieved_docs[0].page_content)

# ----------------------------------------------------------------------------------------------------------------
# FAISS
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

db = FAISS.from_documents(texts, embeddings_model)
retriever = db.as_retriever()
# retriever = db.as_retriever(search_type="mmr")  # 最大边际相关性搜索
# retriever = db.as_retriever(search_kwargs={"k": 1})
docs = retriever.get_relevant_documents("what did he say about ketanji brown jackson")
 
# ----------------------------------------------------------------------------------------------------------------
# TF-IDF检索器
retriever = TFIDFRetriever.from_texts(["foo", "bar", "world", "hello", "foo bar"])
result = retriever.get_relevant_documents("foo")
 
 
#!/usr/bin/env python
# -*- coding = utf-8 -*-
# @Project : langchang-openai
# @Time : 2023/8/14 09:53
# @Author : Ben Li.
# @File: persist_vector_store.py

# 文本转换为向量的嵌入引擎
from langchain.embeddings import HuggingFaceEmbeddings
# 向量数据库
from langchain.vectorstores import Chroma
# 文档加载器
from langchain.document_loaders import TextLoader, CSVLoader, PyPDFLoader
# 文本拆分
from langchain.text_splitter import RecursiveCharacterTextSplitter

import langchain
import os

# 打开langchain debug
langchain.debug = True
# 本地知识库
knowledge_base_dir = "./content"
# 本地向量库
vector_store_dir = "./vector_store"
# 本地向量模型地址
vector_model_name = "{your local models}"


def persist_vector_store():
    doc = []
    for item in os.listdir(knowledge_base_dir):
        if item.endswith("txt"):
            loader = TextLoader(file_path=os.path.join(knowledge_base_dir, item), encoding="utf-8")
            doc.append(loader.load())
        elif item.endswith("csv"):
            loader = CSVLoader(file_path=os.path.join(knowledge_base_dir, item), encoding="utf-8")
            doc.append(loader.load())
        elif item.endswith("pdf"):
            loader = PyPDFLoader(file_path=os.path.join(knowledge_base_dir, item))
            doc.append(loader.load())
    print("提取文本量：", len(doc))
    # 拆分
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)
    docs = []
    for d in doc:
        docs.append(text_splitter.split_documents(d))
        print("拆分文档数：", len(docs))
    # 准备嵌入引擎
    embeddings = HuggingFaceEmbeddings(model_name=vector_model_name)
    # 向量化
    # 会对 OpenAI 进行 API 调用
    vectordb = Chroma(embedding_function=embeddings, persist_directory=vector_store_dir)
    for d in docs:
        vectordb.add_documents(d)
    # 持久化
    vectordb.persist()


if __name__ == '__main__':
    persist_vector_store()

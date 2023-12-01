#!/usr/bin/env python
# -*- coding = utf-8 -*-
# @Project : langchang-openai
# @Time : 2023/8/19 15:52
# @Author : Ben Li.
# @File: qa_retrive_prompt.py
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains.conversational_retrieval.prompts import QA_PROMPT
import langchain

# 打开langchain debug
langchain.debug = True
# 本地向量库
vector_store_dir = "../knowledge_base/vector_store"
# 本地向量模型地址
vector_model_name = "{your vector model dir}"

# 准备嵌入引擎
embeddings = HuggingFaceEmbeddings(model_name=vector_model_name)
# 向量化
vectordb = Chroma(embedding_function=embeddings, persist_directory=vector_store_dir)


def get_prompt(input):
    docs = vectordb.similarity_search(input, k=4)
    docs = [doc.page_content for doc in docs]

    return QA_PROMPT.format(context=docs, question=input)


if __name__ == '__main__':
    docs = get_prompt("什么是langchain")
    print(type(docs))
    print(docs)

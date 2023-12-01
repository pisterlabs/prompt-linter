#!/usr/bin/env python3
# -*- coding utf-8 -*-

import openai, yaml
import faiss
from llama_index import SimpleDirectoryReader, LangchainEmbedding, GPTFaissIndex, ServiceContext
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from llama_index.node_parser import SimpleNodeParser
from llama_index import QueryMode

"""
通过HuggingFacebEmbedding和sentence-transformers计算向量，并保持到faiss中

模型安装：
pip install sentence-transformers
"""

def get_api_key():
    with open("config.yaml", "r", encoding="utf-8") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        openai.api_key = yaml_data["openai"]["api_key"]


if __name__ == '__main__':
    
    # 计算向量的时候，不使用chatgpt
    openai.api_key = ""

    # 加载文件
    documents = SimpleDirectoryReader('./data/faq/').load_data()

    # 句子拆分
    text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=100, chunk_overlap=20)
    parser = SimpleNodeParser(text_splitter=text_splitter)
    nodes = parser.get_nodes_from_documents(documents)

    # 计算向量
    embed_model = LangchainEmbedding(HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    ))
    service_context = ServiceContext.from_defaults(embed_model=embed_model)

    # 向量放到Faiss中
    dimension = 768
    faiss_index = faiss.IndexFlatIP(dimension)
    index = GPTFaissIndex(nodes=nodes,faiss_index=faiss_index, service_context=service_context)

    # 使用chagpt进行查询
    openai.api_key = get_api_key()

    response = index.query(
        "请问你们海南能发货吗？", 
        mode=QueryMode.EMBEDDING,
        verbose=True, 
    )
    print(response)

    response = index.query(
        "你们用哪些快递公司送货？", 
        mode=QueryMode.EMBEDDING,
        verbose=True, 
    )
    print(response)

    response = index.query(
        "你们的退货政策是怎么样的？", 
        mode=QueryMode.EMBEDDING,
        verbose=True, 
    )
    print(response)

# -*- coding: utf-8 -*-
"""
Created on Wed May 31 11:01:25 2023

@author: shangfr
"""

from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma


def init_vectordb(file_path='data_csv/books_all.csv', collection_name="fables_collection", directory='fables_db'):

    embeddings = OpenAIEmbeddings()
    loader = CSVLoader(file_path, encoding='utf-8')
    docs = loader.load()
    vectordb = Chroma.from_documents(
        collection_name=collection_name,
        documents=docs,
        embedding=embeddings,
        persist_directory=directory)

    vectordb.persist()
    print("Vector DB init success! ")

    return vectordb


def load_vectordb(directory='fables_db', collection_name="fables_collection"):

    embeddings = OpenAIEmbeddings()
    vectordb = Chroma(
        collection_name,
        embeddings,
        directory)
    
    return vectordb


#init_vectordb()
#vectordb = load_vectordb(directory='fables_db')
#retriever = vectordb.as_retriever(search_type="mmr")
#results =retriever.get_relevant_documents("猫和老鼠")[0]
#print(results[0].page_content)
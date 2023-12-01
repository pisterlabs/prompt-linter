#!/usr/bin/env python3
import logging
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All
import time

from constants import *

def init_chain():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = vectorstore.as_retriever()

    llm = GPT4All(model=MODEL_PATH)

    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

class WorkGPT:
    def __init__(self):
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        self.vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
        self.llm = GPT4All(model=MODEL_PATH)

    def get_answer(self, query, sources=4):
        if len(query):
            logging.info(f"Running query: {query} with {sources} sources")
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": sources})
            chain = RetrievalQA.from_chain_type(llm=self.llm, retriever=retriever, return_source_documents=True)
            return chain(query)

    def get_sources(self, query, sources=4):
        if len(query):
            logging.info(f"Getting {sources} sources for query: {query}")
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": sources})
            docs = retriever.get_relevant_documents(query)
            return {"result": None, "source_documents": docs}

def interactive():
    workgpt = WorkGPT()

    # Interactive questions and answers
    while True:
        query = input("\n>: ")

        start = time.time()
        result = workgpt.get_answer(query)
        answer, docs = result['result'], result['source_documents']
        end = time.time()

        print(f"\nAnswer (took {round(end - start, 2)} s.):")
        print(answer)
        print("Sources:")
        for document in docs:
            print(document.metadata['source'])

if __name__ == "__main__":
    interactive()
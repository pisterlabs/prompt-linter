# HOW TO Embedding documents and store them in a vector database for retrieving answers

import os, sys
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter

from llms.llm import azure_openai_embeddings, azure_chat_openai_llm
from langchain.vectorstores import Chroma
import numpy as np
import time


def load_docs():
    # Load PDF
    loaders = [
        # Duplicate documents on purpose - messy data
        PyPDFLoader("./documents/MachineLearning-Lecture01.pdf"),
        PyPDFLoader("./documents/MachineLearning-Lecture01.pdf"),
        PyPDFLoader("./documents/MachineLearning-Lecture02.pdf"),
        PyPDFLoader("./documents/MachineLearning-Lecture03.pdf"),
    ]

    docs = []
    for loader in loaders:
        docs.extend(loader.load())

    # Split
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
    token_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=10)

    splits = token_splitter.split_documents(docs)
    print(splits[0])
    print(len(splits))

    # call from_document method for a bunch of 10 documents
    batch_size = 1
    for i in range(0, len(splits), batch_size):
        batch = splits[i : i + batch_size]
        print(len(batch))
        processed_batch = Chroma.from_documents(
            documents=batch,
            embedding=azure_openai_embeddings(),
            persist_directory="data/chroma/",
        )

        time.sleep(1)

    # # FORCED reduce splits for limit quota exceeded
    # # TODO TRY split and store in batches
    # splits = splits[:10]

    # # embeddings and store data
    # embedding = azure_openai_embeddings()

    # persist_directory = "data/chroma/"
    # vectordb = Chroma.from_documents(
    #     documents=splits, embedding=embedding, persist_directory=persist_directory
    # )

    persist_directory = "data/chroma/"
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=azure_openai_embeddings(),
    )

    print(vectordb._collection.count())


def ask_question():
    embedding = azure_openai_embeddings()

    persist_directory = "data/chroma/"
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

    question = "Does machine learning has a large impact"

    docs = vectordb.similarity_search(question, k=3)
    print(docs)


def compare_embeddings():
    # Embed for comparing sentences using numpy.dot
    embedding = azure_openai_embeddings()

    sentence1 = "i like dogs"
    sentence2 = "i like canines"
    sentence3 = "the weather is ugly outside"

    embedding1 = embedding.embed_query(sentence1)
    embedding2 = embedding.embed_query(sentence2)
    embedding3 = embedding.embed_query(sentence3)

    compare1 = np.dot(embedding1, embedding2)
    compare2 = np.dot(embedding1, embedding3)
    compare3 = np.dot(embedding2, embedding3)

    print(compare1)
    print(compare2)
    print(compare3)


load_docs()

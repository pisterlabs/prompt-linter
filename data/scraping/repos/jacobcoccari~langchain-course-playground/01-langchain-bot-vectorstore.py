# pip install "unstructured[md]"
# pip install unstructured
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    Language,
)
from langchain.vectorstores import Chroma
from langchain.document_loaders import DirectoryLoader
import pickle
import os
from dotenv import load_dotenv
import time

load_dotenv()

embedding_function = OpenAIEmbeddings()

character_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
)

db = Chroma(
    embedding_function=embedding_function,
    persist_directory="./11-Langchain-Bot/langchain_documents_db/",
)

python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=1000,
    chunk_overlap=100,
)


def read_documentation():
    new_memory_load = pickle.loads(
        open("./11-Langchain-Bot/langchain_documents.pkl", "rb").read()
    )
    # print(new_memory_load)

    docs = character_text_splitter.split_documents(new_memory_load)
    for doc in docs:
        print(doc)
        db.add_documents([doc])
        print("+")
        time.sleep(0.001)
        db.persist()


def read_source_code():
    loader = DirectoryLoader(
        "./11-Langchain-Bot/langchain",
        glob="**/*.py",
        # loader_cls=python_splitter,
        show_progress=True,
    )
    docs = loader.load()
    # print(docs)
    python_docs = python_splitter.split_documents(docs)
    for doc in python_docs:
        doc.metadata["source"] = doc.metadata["source"].replace(
            "11-Langchain-Bot", "https://github.com/langchain-ai/langchain"
        )
        print(doc.metadata["source"])
        db.add_documents([doc])
        print("+")
        time.sleep(0.001)
        db.persist()


def main():
    read_documentation()
    read_source_code()


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
from langchain.document_loaders import  UnstructuredFileLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import nltk
import openai
from dotenv import load_dotenv
import os
load_dotenv()

nltk.download('punkt')

print('Loading global variables')
# Load Langchain variables
openai.api_key = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings()

llm = OpenAI(temperature=0)

vectorstore_location = './docs/'

text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=25)

print('base_formatter function')

def check_file(file_path):
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load()
    print(docs)
    return docs

def base_formatter(docs):
    print('formatting')
    print(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" +
                                   d.page_content for i, d in enumerate(docs)]))
    return f"\n{'-' * 100}\n".join(
        [f"Document {i + 1}:\n\n{d.page_content}" for i, d in enumerate(docs)]
    )


print('loading check_file function 43')
# Check if the files are valid




def create_mass_embedding(folder_path):
    print('creating mass embedding')
    if not os.path.exists(folder_path):
        folder_path = 'docs/empty'
        result = "Folder does not exist"
        print(result)
        return
    else:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            result = create_embedding(file_path, filename)
            print(f"Embedding created for {filename}: {result}")
            with open('./docs/embed_index.txt', 'a') as f:
                f.write(f"{os.path.join(folder_path, file_path)}\n")
            print(f"Embedding created for {filename}: {result}")

        return result


print('create_embedding function')
# Embed a single embedding


def create_embedding(file_path, optional_arg="metadata"):
    print('creating embedding')
    data = check_file(file_path)
    metadata = optional_arg
    if metadata:
        meta = metadata
    else:
        meta = 'file_path'

    vectordb = Chroma.from_documents(
        documents=data, metadata=meta, embedding=embeddings, persist_directory=vectorstore_location)
    vectordb.persist()
    print(data)
    return "Embedding created"


print('load_vector_store_docs function')


def load_vector_store_docs():
    print('running load_vector_store_docs')
    docs = Chroma(persist_directory=vectorstore_location,
                  embedding_function=embeddings)
    print(docs)
    return docs


print('memory_search function')
# Query the database and pass the info to chatgpt for response

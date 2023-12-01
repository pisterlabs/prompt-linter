#!/usr/bin/env python3
import os
import time
from langchain.llms import GPT4All
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

INPUT_DOCUMENT_PATH = "stellarcorp.txt"
PERSIST_DIRECTORY = "db"
MODEL_PATH = "models/ggml-gpt4all-j-v1.3-groovy.bin"
EMBEDDINGS_MODEL_NAME = "all-MiniLM-L6-v2"

# function to return loaded document with TextLoader
def load_document(file_path: str) -> Document:
    loader = TextLoader(file_path)
    return loader.load()

# initialize the HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)

# load the language model
llm = GPT4All(model=MODEL_PATH, verbose=False, temp=0.2)

# if no existing vectorstore for the document, create it
if not os.path.exists(os.path.join(PERSIST_DIRECTORY, 'index')):
    # load company document
    file = load_document(INPUT_DOCUMENT_PATH)

    # split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    texts = text_splitter.split_documents(file)

    # create the vectorstore
    db = Chroma.from_documents(texts, embeddings, persist_directory=PERSIST_DIRECTORY)
else:
    # initialize existing vectorstore
    db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)

# expose this index in a retriever interface
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# create a chain to answer questions 
qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

# start querying the llm
while True:
    query = input("\nQuery: ")

    # break condition
    if query == "exit":
        break
    if not query.strip():
        continue

    start_time = time.time()
    response = qa(query)
    elapsed_time = round(time.time() - start_time, 2)
    answer = response['result']
    
    print("\n\n> Question:", query)
    print(f"\n> Answer (took {elapsed_time} s.):", answer)

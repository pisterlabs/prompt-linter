import os
import shutil
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


def db_from_docs(docs):
    embed_model = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-mpnet-base-v2')
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, chunk_overlap=200)
    texts = text_splitter.split_documents(docs)
    db_loc = './db '+docs[0].metadata['source'].split('/')[-1]
    if os.path.isdir(db_loc):
        try:
            shutil.rmtree(db_loc)
        except Exception as e:
            print("Unable to remove dir", e)
    db = Chroma.from_documents(
        documents=texts, embedding=embed_model, persist_directory=db_loc)
    retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 4})
    return retriever


def db_from_dir(db_dir='./db'):
    embed_model = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-mpnet-base-v2')
    db = Chroma(persist_directory=db_dir, embedding_function=embed_model)
    retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 4})
    return retriever

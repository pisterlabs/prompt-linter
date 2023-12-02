from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from .pdf_loader import docs
from .openai_vars import load_openai_key
import os

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 150
)

load_openai_key()
splits = text_splitter.split_documents(docs)
embedding = OpenAIEmbeddings()

cwd = os.path.dirname(os.path.realpath(__file__))
persist_directory = os.path.join(cwd,'docs','chroma')
file_path = os.path.join(cwd,'docs','chroma','chroma-embeddings.parquet')

if not os.path.exists(file_path):
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        persist_directory=persist_directory
    )
    vectordb.persist()
else:
    print('loading existing db...')
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)


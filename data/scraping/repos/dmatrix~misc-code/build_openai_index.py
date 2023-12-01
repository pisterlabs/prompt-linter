import os
import time
from random import randint
from pathlib import Path

from langchain.document_loaders import PyPDFLoader 
from langchain.embeddings import OpenAIEmbeddings 
from langchain.vectorstores import Chroma 

import ray

CHUNKS = 100

def split_pages(pages, n): 
    for i in range(0, len(pages), n):
        yield pages[i:i + n]

@ray.remote
def create_embeddings(shard: str, vector_db_path: str, open_ai_key: str):
    os.environ["OPENAI_API_KEY"] = open_ai_key
    embeddings = OpenAIEmbeddings()
    persist_dir = f"{vector_db_path}"
    vectordb = Chroma.from_documents(shard, embedding=embeddings,
                                 persist_directory=persist_dir)
    vectordb.persist()
    return len(shard)
    
def create_index(document: str, vector_db_path: str, open_ai_key: str) -> None:

    loader = PyPDFLoader(document)
    pages = loader.load_and_split()
    # chunks = list(split_pages(pages, CHUNKS))

    # We could use Ray Distribute creating vector embeddings indexes
    # here we using only one pdf but we could send a list of pdfs
    vectordbs = ray.get(create_embeddings.remote(pages, vector_db_path, open_ai_key))
    print(f"Embeddings created from {vectordbs} pages")

if __name__ == "__main__":
    KEY = "you_key"
    document = Path(Path.cwd(), "hai_ai_index_report.pdf").as_posix()
    vector_db_path = Path(Path.cwd(), "vector_oai_db").as_posix()
    index_path = Path(vector_db_path, "index")
    if not index_path.exists():
        start = time.time()
        create_index(document=document, vector_db_path=vector_db_path, open_ai_key=KEY)
        end = time.time()
        print(f"Time to build index: {end-start:.2f} seconds")
    else: 
        print(f"embeddings already esxist in: {vector_db_path} directory")

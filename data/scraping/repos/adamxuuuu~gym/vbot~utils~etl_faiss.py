import os
import pprint
from typing import Iterator, List, Optional

from tqdm import tqdm
from langchain.vectorstores import FAISS
from langchain.document_loaders import UnstructuredFileLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

from pdf import clean_text

from config import (
    EMBEDDING,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    SEPARATORS
)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, 
    chunk_overlap=CHUNK_OVERLAP,
    separators=SEPARATORS,
    is_separator_regex=True,
    keep_separator=False
)

def load_dir(data_path: str, ftype: Optional[str] = "*.pdf") -> List[Document]:
    return DirectoryLoader(
        path=data_path,
        glob=ftype,
        loader_cls=UnstructuredFileLoader,
        # loader_kwargs={
        #     "mode": "single"
        # },
        show_progress=True
    ).load_and_split(
        text_splitter
    )

def save_local(data_path: str, save_path: str, model: HuggingFaceEmbeddings):
    docs = load_dir(data_path)
    for doc in docs:
        clean_text(doc.page_content)
    vector_db = FAISS.from_documents(
        documents=docs,
        embedding=model,
    )
    vector_db.save_local(save_path)

if __name__ == '__main__':
    # Init Model
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING)
    # Save
    model_base = os.path.basename(EMBEDDING)
    db_path = f'./faiss/cs{CHUNK_SIZE}/{model_base}'
    data_path = './data/test'
    save_local(data_path, db_path, embeddings)
    # Query
    # vector_db = FAISS.load_local(db_path, embeddings)
    # hits = vector_db.search('who is the contact person for environment friendly policy', search_type='mmr')
    # for hit in hits:
    #     pprint.pprint(hit.page_content)
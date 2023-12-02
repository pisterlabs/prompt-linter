from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter

from src.codevecdb.config.Config import conf
from src.codevecdb.db.milvus_vectordb import insert_doc_db


def parse_text_and_insert(text):
    text_splitter = CharacterTextSplitter(chunk_size=conf.chunk_size, chunk_overlap=conf.chunk_overlap)
    metadata = {"source": "upload/1"}
    doc = [Document(page_content=text, metadata=metadata)]
    docs_list = text_splitter.split_documents(doc)
    insert_doc_db(docs_list)
    return 



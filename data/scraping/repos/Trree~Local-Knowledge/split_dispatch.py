import os

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

from src.codevecdb.config.Config import conf
from src.codevecdb.db.milvus_vectordb import insert_doc_db


def split_file_to_chunks(file):
    print(file)
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    file.save('uploads/' + file.filename)  # 保存文件到指定路径
    loader = TextLoader(os.path.join('uploads', file.filename), encoding='utf8')
    documents = loader.load()
        
    text_splitter = CharacterTextSplitter(chunk_size=conf.chunk_size, chunk_overlap=conf.chunk_overlap)
    docsList = text_splitter.split_documents(documents)

    vector_store = insert_doc_db(docsList)
    print(vector_store)


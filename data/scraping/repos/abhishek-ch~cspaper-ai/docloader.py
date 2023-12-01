from langchain.vectorstores import Qdrant
from paperai.config import *
from langchain.embeddings.openai import OpenAIEmbeddings
from paperai.llm import ChatLLM
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client.http import models

def upload(model:str="gpt-35-turbo") -> None:
    chatllm = ChatLLM(model=model)
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002",chunk_size=1)
    loader = PyPDFDirectoryLoader(pdf_uploadpath)
    # Load the PDF document
    documents = loader.load()  
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50,separators=["\n\n", ""])
    docs = text_splitter.split_documents(documents)
    print(f"Doc Loaded {docs[0]}")
    print(f"db_persistent_path {db_persistent_path}\n collection_name {collection_name}")
    qdrant_doc = Qdrant.from_documents(
        docs, embeddings, 
        path=db_persistent_path,
        collection_name=collection_name
    )
    qdrant_doc = None



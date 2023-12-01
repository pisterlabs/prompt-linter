from langchain.document_loaders import WebBaseLoader, PyPDFLoader, DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import FAISS, Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import SVMRetriever

import numpy as np

FAISS_PATH = "vectorstores/db_faiss"
CHROMA_PATH = "vectorstores/db_chroma"

def get_index_vectorstore_wiki_nyc(embed_model):
    # load the Wikipedia page and create index
    loader = WebBaseLoader("https://en.wikipedia.org/wiki/New_York_City") # pip install bs4
    index = VectorstoreIndexCreator(
        vectorstore_cls=FAISS,
        embedding=embed_model,
        # text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        # vectorstore_kwargs={ "persist_directory": "/vectorstore"},
    ).from_loaders([loader]) 
    return index


def dataset_to_texts(data):
    data_pd = data.to_pandas()
    texts = data_pd['chunk'].to_numpy()
    return texts

# create vector database
def create_local_faiss_vector_database(texts, embeddings, DB_PATH, data=None):
    """
    Create a local vector database from a list of texts and an embedding model.
    you can change the input from texts or data
    """
    # Loader for PDFs
    # loader = DirectoryLoader(DATA_PATH, glob = '*.pdf', loader_cls= PyPDFLoader)
    # documents = loader.load()
    # text_splitter = RecursiveCharacterTextSplitter (chunk_size = 500, chunk_overlap = 50)
    # texts = text_splitter.split_documents(documents)

    # text splitter for dataset
    if data is not None:
        texts = dataset_to_texts(data)
        text_splitter = RecursiveCharacterTextSplitter (chunk_size = 500, chunk_overlap = 50)
        texts = text_splitter.split_texts(texts)
    
    # db = FAISS.from_texts(texts, embeddings)
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_PATH)

def load_local_faiss_vector_database(embeddings):
    """
    Load a local vector database from a list of texts and an embedding model.
    """
    db = FAISS.load_local(FAISS_PATH, embeddings)
    return db

    
def create_chroma_db(documents, embeddings):
    vectorstore = Chroma.from_documents(documents=documents, 
                                        embedding=embeddings,
                                        persist_directory=CHROMA_PATH)
    return vectorstore

def load_chroma_db(embeddings):
    vectorstore = Chroma(persist_directory=CHROMA_PATH, 
                         embedding_function=embeddings)
    return vectorstore
    

def similarity_search_doc(db, query):
    """
    Ref:
    https://github.com/JayZeeDesign/Knowledgebase-embedding/blob/main/app.py
    """
    similar_response = db.similarity_search(query, k=3)

    page_contents_array = get_content_from_similarity_search(similar_response)

    print(len(page_contents_array))

    return page_contents_array


def svm_similarity_search_doc(documents, query, embed_model):

    svm_retriever = SVMRetriever.from_documents(documents=documents,
                                                embeddings=embed_model,
                                                k = 3,
                                                relevancy_threshold = 0.3)
    docs_svm=svm_retriever.get_relevant_documents(query)
    docs_svm_list = get_content_from_similarity_search(docs_svm)
    len(docs_svm)
    return docs_svm_list

def get_content_from_similarity_search(similar_response):
    page_contents_array = [doc.page_content for doc in similar_response]
    return page_contents_array


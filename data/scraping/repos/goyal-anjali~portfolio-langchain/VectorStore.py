from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.vectorstores import Chroma
import os

def load_data(folderlocation):
    locationlist = os.listdir(folderlocation)
    loaders = []
    for location in locationlist:
        loaders.append(UnstructuredHTMLLoader(os.path.join(folderlocation, location)))
    docs = []
    for l in loaders:
        docs.extend(l.load())
    return docs

def split_data(docs):
    text_splitter = CharacterTextSplitter(chunk_size=2460, chunk_overlap=200)
    documents = text_splitter.split_documents(docs)
    return documents
    
def store_data(documents, embeddings):
    # The below model is free and better than the one we used currently, but it takes a lot of time.
    # To use it, uncomment the below 2 lines and comment the line after it.
    # from langchain.embeddings import HuggingFaceInstructEmbeddings
    # embeddings = HuggingFaceInstructEmbeddings(model_name = "hkunlp/instructor-xl")
    vectordb = Chroma.from_documents(documents, embedding=embeddings, persist_directory="./chroma_db");
    vectordb.persist();
    
def create_dataset(locationlist, embedding):
    data = load_data(locationlist)
    documents = split_data(data)
    store_data(documents, embedding)

def get_dataset(embeddings):
    datastore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    return datastore
    
def getDataRetriever(db):
    return db.as_retriever()
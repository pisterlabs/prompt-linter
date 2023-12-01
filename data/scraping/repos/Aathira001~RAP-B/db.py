from django.conf import settings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

docs_path = f'{settings.BASE_DIR}/data/media/'
db_path = f'{settings.BASE_DIR}/data/db/'
embeddings_model = 'sentence-transformers/all-MiniLM-L6-v2'


def load_docs(file_name):
    loader = PyPDFLoader(docs_path + file_name)
    docs = loader.load()
    return docs

def get_data(file_name):
    docs = load_docs(file_name)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    data = text_splitter.split_documents(docs)
    return data

def get_embeddings(embeddings_model):
    return HuggingFaceEmbeddings(model_name = embeddings_model)

def create_chroma_db(file_name):
    data = get_data(file_name)
    embeddings = get_embeddings(embeddings_model)
    db = Chroma(persist_directory=db_path, embedding_function=embeddings)
    if not db.get()['documents']:
        chroma_db = Chroma.from_documents(data, embeddings, persist_directory=db_path)
        chroma_db.persist()
        chroma_db = None
    else:
        print("*************************************************")
        print(f"Adding new data {file_name}")
        print("*************************************************")
        db.add_documents(data)
        db.persist()
        db = None
    
def get_chroma_db_retriever():
    embeddings = get_embeddings(embeddings_model)
    chroma_db = Chroma(persist_directory=db_path,
                  embedding_function=embeddings
                  )
    return chroma_db.as_retriever(search_kwargs={"k": 1})
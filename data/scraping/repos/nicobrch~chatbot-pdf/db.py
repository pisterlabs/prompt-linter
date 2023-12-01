from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Milvus
from dotenv import load_dotenv

load_dotenv()

# CARGAR DOCUMENTOS
loader = PyPDFDirectoryLoader("docs/")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)  # milvus
embeddings = OpenAIEmbeddings()


def milvusDB():  # BASE DE DATOS MILVUS
    db = Milvus.from_documents(
        docs,
        embeddings,
        connection_args={
            "host": "localhost",
            "port": "19530"
        },
    )
    return db

import os
import pinecone 
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import tomllib

load_dotenv()

def load_local_environment_variables():
    with open(".streamlit/secrets.toml", "rb") as f:
        return tomllib.load(f)

secrets = load_local_environment_variables()

PINECONE_API_KEY = secrets['PINECONE_API_KEY']
PINECONE_ENV = secrets['PINECONE_ENVIRONMENT']
PINECONE_INDEX = secrets['PINECONE_INDEX']
OPENAI_API_KEY = secrets['OPENAI_API_KEY']

def doc_preprocessing():
    loader = DirectoryLoader(
        'data/',
        glob='**/*.pdf',     # only the PDFs
        show_progress=True
    )

    docs = loader.load()

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)

    docs_split = text_splitter.split_documents(docs)
    return docs_split

def create_embeddings():
    # we use the openAI embedding model
    embeddings = OpenAIEmbeddings(
       openai_api_key=OPENAI_API_KEY
    )

    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENV
    )

    pinecone.describe_index(PINECONE_INDEX)

    docs_split = doc_preprocessing()
    print(docs_split)
    doc_db = Pinecone.from_documents(
        docs_split, 
        embeddings, 
        index_name=PINECONE_INDEX
    )
    return doc_db

def main():
    create_embeddings()
    print("Embeddings completed")

if __name__ == "__main__":
    main()
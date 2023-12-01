import config
import os
import openai


from dotenv import load_dotenv, find_dotenv
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter


def upload():

    try:
        _ = load_dotenv(find_dotenv())
    except Exception as e:
        print(e)
    openai.api_key = os.getenv("OPENAI_API_KEY")

    DATA_PATH = "data/"
    TXT_PATH = DATA_PATH + 'txts/'
    PDF_PATH = DATA_PATH + 'pdfs/'

    # Embedding av text dokumenter
    loader = DirectoryLoader(TXT_PATH, glob="**/*.txt",
                             loader_cls=TextLoader, use_multithreading=True)
    documents = loader.load()

    # Embedding av pdf dokumenter
    # loader = DirectoryLoader(PDF_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
    # documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)

    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(model_kwargs={"device": "mps"})

    VECTORDB = 'vectordb'
    persist_directory = f"embeddings/{VECTORDB}/"

    vectordb = Chroma.from_documents(
        documents=texts, embedding=embeddings, persist_directory=persist_directory
    )
    vectordb.persist()


def main():
    upload()


if __name__ == '__main__':
    main()

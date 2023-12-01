import os
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone

from consts import INDEX_NAME

pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT"],
)


def ingest_docs() -> None:
    loader = PyPDFDirectoryLoader("data/")
    raw_documents = loader.load()
    print(f"loaded {len(raw_documents) } documents")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", " ", ""],
    )
    documents = text_splitter.split_documents(documents=raw_documents)

    print(f"Splitted into {len(documents)} chunks")

    print(f"Going to insert {len(documents)} to Pinecone")
    embeddings = OpenAIEmbeddings(disallowed_special=())
    Pinecone.from_documents(
        documents=documents, embedding=embeddings, index_name=INDEX_NAME
    )
    print("Added to Pinecone vectorstore vectors")


if __name__ == "__main__":
    ingest_docs()

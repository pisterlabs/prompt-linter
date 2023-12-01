import os
from langchain.document_loaders import (
    ReadTheDocsLoader,
)  # this help to create github repo.
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone, FAISS
import pinecone
from dotenv import load_dotenv


def ingest_docs():
    loader = ReadTheDocsLoader("langchain-docs/api.python.langchain.com/en/latest")
    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", " ", ""]
    )

    documents = text_splitter.split_documents(raw_documents)
    print(f"splitted into {len(documents)} chucks")

    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    print(f"going to insert {len(documents)} to pinecone")

    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

    print(f"Going to add {len(documents)} to Pinecone")

    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENVIRONMENT"),
    )

    Pinecone.from_documents(
        documents, embedding=embeddings, index_name=os.getenv("PINECONE_INDEX_NAME")
    )

    # vectorstore = FAISS.from_documents(documents, embeddings)
    # vectorstore.save_local("faiss_index_book")
    # new_vectorstore = FAISS.load_local("faiss_index_book", embeddings)
    # print(new_vectorstore)
    print("****Loading to vectorestore done ***")


if __name__ == "__main__":
    load_dotenv()

    ingest_docs()

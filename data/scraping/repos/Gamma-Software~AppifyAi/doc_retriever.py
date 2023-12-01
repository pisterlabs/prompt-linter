import shutil
import os
import sys
from typing import List


import streamlit as st
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import Language
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers.language import LanguageParser
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb.utils import embedding_functions
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import UnstructuredMarkdownLoader
import chromadb
import uuid
from chromadb.config import Settings
import tiktoken


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def num_tokens_from_string_list(docs: List[Document], encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = 0
    max_tokens = 0
    for doc in docs:
        if len(encoding.encode(doc.page_content)) > max_tokens:
            max_tokens = len(encoding.encode(doc.page_content))
        num_tokens += len(encoding.encode(doc.page_content))
    return num_tokens, max_tokens


def load_streamlit_doc_retriever(
    openai_api_key: str,
    chroma_server_host="localhost",
    chroma_server_port="8000",
    mode="docker",
) -> VectorStoreRetriever:
    if openai_api_key is None:
        raise Exception("Please provide an OpenAI API key.")

    # Check if the Chroma database exists
    if mode == "local":
        if not os.path.exists(".doc_db/streamlit_chroma_db"):
            raise Exception(
                "The Chroma database for Streamlit does not exist. "
                "Please run the script `doc_retriever.py` to create it."
            )

    # load from disk
    if mode == "local":
        retriever = Chroma(
            persist_directory=".doc_db/streamlit_chroma_db",
            embedding_function=OpenAIEmbeddings(openai_api_key=openai_api_key),
        ).as_retriever()
    if mode == "docker":
        client = chromadb.Client(
            Settings(
                chroma_api_impl="rest",
                chroma_server_host=chroma_server_host,
                chroma_server_http_port=chroma_server_port,
            )
        )

        # tell LangChain to use our client and collection name
        retriever = Chroma(
            client=client,
            collection_name="streamlit_doc",
            embedding_function=OpenAIEmbeddings(openai_api_key=openai_api_key),
        ).as_retriever()

    retriever.search_kwargs["distance_metric"] = "cos"
    retriever.search_kwargs["fetch_k"] = 4
    retriever.search_kwargs["maximal_marginal_relevance"] = True
    retriever.search_kwargs["k"] = 4
    return retriever


def is_docker_container_running(container_name: str) -> bool:
    """Returns True if the Docker container is running, False otherwise."""
    if os.system(f"docker ps | grep {container_name} > /dev/null") == 0:
        return True
    print(
        f"The Docker container {container_name} is not running."
        "Please run `docker-compose up -d` to start it."
    )
    return False


def generate_retriever(
    openai_api_key: str,
    chroma_server_host="localhost",
    chroma_server_port="8000",
    mode="docker",
):
    if mode == "local":
        # Check if the Chroma database exists
        if os.path.exists(".doc_db/streamlit_chroma_db"):
            try:
                choice = input(
                    "The Chroma database for Streamlit already exists. "
                    "Press Yes to delete it and create a new one. "
                    "Press Ctrl+C or enter anything else to cancel."
                )
                if choice in ["y", "Y", "yes", "Yes", "YES"]:
                    shutil.rmtree(".doc_db/streamlit_chroma_db")
                else:
                    raise KeyboardInterrupt
            except KeyboardInterrupt:
                print("Cancelled.")
                exit(0)

    print("=== Clone the latest version of the Streamlit Doc repo...")
    if os.path.exists("docs"):
        shutil.rmtree("docs")
    os.system(
        "git clone https://github.com/streamlit/docs.git --branch main --depth 1 docs"
    )

    print(
        "=== Create documents containing the Streamlit code base and split them into chunks..."
    )

    steamlit_doc_loader = DirectoryLoader(
        "docs",
        glob="content/**/*.md",
        loader_cls=UnstructuredMarkdownLoader,
        show_progress=True,
        use_multithreading=True,
    )
    steamlit_doc = steamlit_doc_loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=2000,
        chunk_overlap=50,
    )
    steamlit_doc_splitted = text_splitter.split_documents(steamlit_doc)
    steamlit_doc_splitted

    # Load python exemple source code
    loader = GenericLoader.from_filesystem(
        "docs/",
        glob="python/api-examples-source/**/*.py",
        suffixes=[".py"],
        parser=LanguageParser(language=Language.PYTHON),
        show_progress=True,
    )
    streamlit_code_example_doc = loader.load()

    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=500, chunk_overlap=50
    )
    streamlit_code_example_doc_splitted = python_splitter.split_documents(
        streamlit_code_example_doc
    )

    token, max_token = num_tokens_from_string(steamlit_doc_splitted, "cl100k_base")
    print(
        f"Number of tokens in source code: {token}, max "
        f"tokens in source code: {max_token}, price: {token * 0.0001/1000}"
    )
    token, max_token = num_tokens_from_string(
        streamlit_code_example_doc_splitted, "cl100k_base"
    )
    print(
        f"Number of tokens in source code: {token}, max "
        f"tokens in source code: {max_token}, price: {token * 0.0001/1000}"
    )

    print("=== Create embeddings and save them into Chroma Database for later use...")
    embeddings_model = OpenAIEmbeddings(openai_api_key=openai_api_key)

    docs = steamlit_doc_splitted + streamlit_code_example_doc_splitted
    if mode == "local":
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings_model,
            persist_directory=".doc_db/streamlit_chroma_db",
        )
        vectorstore.persist()
    if mode == "docker":
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=openai_api_key,  # Replace with your own OpenAI API key
            model_name="text-embedding-ada-002",
        )
        client = chromadb.Client(
            Settings(
                chroma_api_impl="rest",
                chroma_server_host=chroma_server_host,
                chroma_server_http_port=chroma_server_port,
            )
        )
        print(client.heartbeat())
        client.reset()  # resets the databas
        collection = client.create_collection(
            "streamlit_doc", embedding_function=openai_ef
        )
        for doc in docs:
            collection.add(
                ids=[str(uuid.uuid1())],
                metadatas=doc.metadata,
                documents=doc.page_content,
            )

    print("=== Remove Streamlit codesource.")
    if os.path.exists("streamlit"):
        shutil.rmtree("streamlit")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("OpenAI API key is missing! Please add it in argument.")
        print("Or you are missing the mode 'docker' or 'local'.")
        exit(1)
    openai_api_key = sys.argv[1]
    generate_retriever(
        openai_api_key,
        mode=sys.argv[2],
        chroma_server_host=st.secrets["chroma"]["host"],
        chroma_server_port=st.secrets["chroma"]["port"],
    )

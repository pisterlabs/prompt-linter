# -*- coding: utf-8 -*-

# This requires Docker containers for Chroma, Redis and Qdrant to be running.

# Importing required libraries
import argparse
import os
import sys
import time
import constants
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.redis import Redis
from langchain.vectorstores.chroma import Chroma
from qdrant_client import QdrantClient
from langchain.vectorstores.qdrant import Qdrant

from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain import hub
from langchain.document_loaders import DirectoryLoader, PyPDFLoader

# Set OpenAI API key from constants file
os.environ["OPENAI_API_KEY"] = constants.OPENAI_API_KEY


def process_llm_response(llm_response):
    # Outputting the answer or debugging information in case of an error
    if "result" in llm_response:
        print(f"Answer: {llm_response['result']}")
    else:
        print(
            "Result key not found in the returned object. Here's the full object for debugging:"
        )
        print(llm_response)

    print("\n\nSources:")
    for source in llm_response["source_documents"]:
        print(source.metadata["source"])


def main():
    while True:
        # Prompting user for vector storage selection
        choice = input(
            "Select vector storage: 1 for Chroma, 2 for Redis, 3 for Qdrant: "
        )

        # Loading documents from a specified directory
        loader = DirectoryLoader(
            "./whitepapers/", glob="./*.pdf", loader_cls=PyPDFLoader
        )
        documents = loader.load()
        # Splitting documents into manageable text chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=100
        )
        all_splits = text_splitter.split_documents(documents)
        print(
            f"Processed {len(documents)} documents split into {len(all_splits)} chunks"
        )

        # Initializing vector storage based on user choice
        start_time = time.time()
        if choice == "1":
            # Configuring Chroma vectorstore settings
            from chromadb.config import Settings


            try:
                chroma_settings = Settings(
                    chroma_server_host="http://asdf",
                    chroma_server_http_port="8000",
                    chroma_server_ssl_enabled=False,
                )
                # chroma_settings = Settings(
                #     chroma_server_host="34.141.229.240",
                #     chroma_server_http_port="8080",
                #     chroma_server_ssl_enabled=False,
                # )
                # Initializing Chroma vectorstore with document embeddings
                vectorstore = Chroma.from_documents(
                    documents=all_splits,
                    embedding=OpenAIEmbeddings(),
                    persist_directory="vectorstore_db",
                    client_settings=chroma_settings,
                )
                # Example operation to force connection
                some_test_document_embedding = vectorstore.get("vectorstore_db")
                print(some_test_document_embedding)
            except Exception as e:
                print(f"Failed to initialize Chroma vectorstore: {e}")
                continue

        elif choice == "2":
            # Initializing Redis vectorstore with document embeddings
            try:
                vectorstore = Redis.from_documents(
                    documents=all_splits,
                    embedding=OpenAIEmbeddings(),
                    # redis_url="redis://localhost:6379",
                    redis_url="redis://asdf",
                )
            except Exception as e:
                print(f"Failed to initialize Redis vectorstore: {e}")
                continue

        elif choice == "3":
            try:
                # !!! You will need this the first time you run this script as the collection needs to be created !!!
                # Setting up QdrantClient and creating a collection for vector storage
                # from qdrant_client.http.models import Distance, VectorParams
                # try:
                #     qdrant_client = QdrantClient(url="34.141.229.240", port=6333)
                #     qdrant_client.create_collection(
                #         collection_name="test_collection",
                #         vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
                #     )
                # except Exception as e:
                #     print(f"Failed to initialize Qdrant client or create collection: {e}")
                #     continue

                # Initializing Qdrant vectorstore with document embeddings
                url = "http://localhost:6333"
                # url = "http://34.141.229.240:6333"
                vectorstore = Qdrant.from_documents(
                    collection_name="test_collection",
                    embedding=OpenAIEmbeddings(),
                    documents=all_splits,
                    url=url,
                )
            except Exception as e:
                print(f"Failed to initialize Qdrant vectorstore: {e}")
                continue
        else:
            # Handling invalid input for vector storage selection
            print(
                "Invalid choice. Please select 1 for Chroma, 2 for Redis, or 3 for Qdrant."
            )
            continue

        # Loading the Language Model with a callback manager
        llm = Ollama(
            model="openhermes2-mistral",
            verbose=True,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        )

        # Setting up a QA Chain with a specific prompt
        QA_CHAIN_PROMPT = hub.pull("rlm/rag-prompt-llama")
        qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=vectorstore.as_retriever(),
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
            return_source_documents=True,
        )

        end_time = time.time()
        # Calculating and displaying the time taken for setup
        time_taken_ms = (
            end_time - start_time
        ) * 1000  # Convert seconds to milliseconds
        print(f"Time taken: {time_taken_ms} milliseconds")

        # Asking a question and measuring response time
        question = "What is the toolformer?"

        try:
            # Getting the answer using the QA chain
            result = process_llm_response(qa_chain({"query": question}))

        except Exception as e:
            # Handling exceptions during the QA process
            end_time = time.time()
            print("An error occurred:", e)
            print("Here's the partial result for debugging:")
            print(result)


if __name__ == "__main__":
    main()

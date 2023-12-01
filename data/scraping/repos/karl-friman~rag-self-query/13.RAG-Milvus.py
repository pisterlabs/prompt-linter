# -*- coding: utf-8 -*-

# This requires Docker containers for Chroma, Redis and Qdrant to be running.
import os, constants

from langchain.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain import hub
from langchain.document_loaders import DirectoryLoader, PyPDFLoader

from langchain.vectorstores import Milvus

os.environ["OPENAI_API_KEY"] = constants.OPENAI_API_KEY

def main():
    # Loading documents from a specified directory
    loader = DirectoryLoader(
        "./data/", glob="./*.pdf", loader_cls=PyPDFLoader
    )
    documents = loader.load()
    # Splitting documents into manageable text chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=100
    )
    all_splits = text_splitter.split_documents(documents)
    print(
        f"RecursiveCharacterTextSplitter from Langchain:\nProcessed {len(documents)} documents split into {len(all_splits)} chunks"
    )

    try:
        # This is the first run with new data
        vectorstore = Milvus.from_documents(
            documents=all_splits,
            embedding=OpenAIEmbeddings(),
            # connection_args={"host": "127.0.0.1", "port": "19530"},
            # connection_args={"host": "localhost", "port": "19530"},
            connection_args={"host": "34.141.233.82", "port": "19530"},
        )
        # This is how to load an existing collection
        # vectorstore = Milvus(
        #     # If you have another collection than what langchain creates, you can specify it here
        #     # collection_name="collection_1",
        #     embedding_function=OpenAIEmbeddings(),
        #     connection_args={"host": "localhost", "port": "19530"},
        # )
                
    except Exception as e:
        print(f"Failed to initialize vectorstore: {e}")

    # Loading the Language Model with a callback manager
    llm = Ollama(
        model="openhermes2-mistral",
        verbose=True,
        temperature=0.0,
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

    def process_llm_response(llm_response):
        print("\n\nSources:")
        for source in llm_response["source_documents"]:
            print(source.metadata["source"])
            
    # Asking a question on the Toolformer PDF
    question = "What is the conclusion summary for the Toolformer whitepaper?"
    print(f"Question: {question}")
    process_llm_response(qa_chain({"query": question}))

    # Asking a question based on the other PDF.
    question = "What is the name of the cat?"
    print(f"Question: {question}")
    process_llm_response(qa_chain({"query": question}))

if __name__ == "__main__":
    main()

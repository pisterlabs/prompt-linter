# -*- coding: utf-8 -*-

import argparse
import os
import constants
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.redis import Redis
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain import hub
from langchain.chains import RetrievalQA

# Set OpenAI API key from constants
os.environ["OPENAI_API_KEY"] = constants.OPENAI_API_KEY


def main():
    # Set up argument parser for command line options
    parser = argparse.ArgumentParser(
        description="Process a URL to perform retrieval-based QA."
    )
    parser.add_argument("--url", type=str, required=True, help="The URL to process.")
    args = parser.parse_args()
    url = args.url
    print(f"Using URL: {url}")

    # Load and split webpage text
    loader = WebBaseLoader(url)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    all_splits = text_splitter.split_documents(data)
    print(f"Split into {len(all_splits)} chunks")

    # Setup Redis Vectorstore
    vectorstore = Redis.from_documents(
        documents=all_splits,
        embedding=OpenAIEmbeddings(),
        redis_url="redis://localhost:6379",
    )
    print(f"Loaded {len(data)} documents")

    # Load Language Model with streaming output
    llm = Ollama(
        model="openhermes2-mistral",
        verbose=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )
    print(f"Loaded LLM model {llm.model}")

    # Setup QA Chain
    QA_CHAIN_PROMPT = hub.pull("rlm/rag-prompt-llama")
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )

    # Perform QA
    question = f"What are the latest headlines on {url}? Anything about embeddings?"
    result = qa_chain({"query": question})
    # The result can be further processed or displayed here
    # print(result)


if __name__ == "__main__":
    main()

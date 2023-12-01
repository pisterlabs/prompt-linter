#!/usr/bin/env python3

"""
This module provides the entry point for the DocuBot application. It includes functions for building a knowledge base, initializing the application, and running the chatbot. The main function `docubot()` creates an argparse to parse the data directory as a mandatory argument, loads API keys, creates or loads an existing index, and instantiates the QA chatbot. 
"""

from typing import List, TypeVar

T = TypeVar("T")


def chatbot(vector_store):
    """
    A chatbot function that allows users to ask questions and receive answers with citations.

    Args:
        vector_store (list): A list of vectors used for semantic search.

    Returns:
        None
    """
    import time
    from docubot_utils.docubot_utils import answer_question_session, format_citations

    i = 1
    print("Welcome to the chatbot. Type 'quit' or 'exit' to exit")
    chat_history = []
    while True:
        question = input(f"Question #{i}: ")
        if question.lower() in ["quit", "exit"]:
            print("Exiting chatbot")
            time.sleep(2)
            break
        result, chat_history = answer_question_session(
            question, vector_store=vector_store, chat_history=chat_history
        )
        print(f"\nAnswer #{i}: {result['answer']}")
        # Join vector in result['source_document'] into a string seperated by newline character
        citations = format_citations(result["source_documents"])
        print(f"\nCitations #{i}:\n{citations}")
        print(f"\n {'-' * 50} \n")
        i += 1


def build_kb(data_directory: str) -> List[T]:
    """
    Builds a knowledge base by loading documents from the given data directory.

    Args:
        data_directory (str): The path to the directory containing the documents.

    Returns:
        List[T]: A list of chunks, where each chunk contains 512 tokens.
    """
    import os
    from tqdm import tqdm
    from document_loaders.document_loaders import load_document, chunk_data

    list_of_doc_names = []
    # List of supported document types
    extensions = (".pdf", ".docx", ".md", ".txt")
    for root, _, files in os.walk(data_directory):
        for file in files:
            if file.endswith(extensions):
                list_of_doc_names.append(os.path.join(root, file))
            else:
                continue

    # This is the knowledge base that is chunked into 512 tokens per chunk
    data = []
    print(f"Splitting documents into chunks")
    for doc_name in tqdm(list_of_doc_names):
        data.extend(load_document(doc_name))

    print(f"There are {len(data)} pages in the knowledge base")
    chunks = chunk_data(data, chunk_size=512, chunk_overlap=20)
    print(f"These have been split into {len(chunks)} chunks for indexing")
    return chunks


def init():
    """
    Initializes Pinecone API with API key and environment variables.
    """
    from dotenv import load_dotenv, find_dotenv
    import pinecone
    import os

    # Load API keys
    load_dotenv(find_dotenv(), override=True)
    pinecone.init(
        api_key=os.environ.get("PINECONE_API_KEY"),
        environment=os.environ.get("PINECONE_ENV"),
    )


def docubot():
    """
    This function is the entry point of the DocuBot application. It creates an argparse to parse data directory which is a mandatory argument.
    It loads API keys and creates or loads existing index and instantiate QA chatbot.
    """
    import argparse
    import pinecone
    from pinecone_utils.pinecone_utils import (
        create_index,
        create_vector_store,
        fetch_vector_store,
        delete_pinecone_index,
    )
    from langchain.vectorstores import Pinecone

    parser = argparse.ArgumentParser(description="DocuBot")
    parser.add_argument("docs_dir", type=str, help="Path to the documents directory")
    parser.add_argument(
        "--index_name",
        type=str,
        default="docubot-index",
        help="Descriptive name for backend vector db index",
    )
    parser.add_argument(
        "--reload",
        type=str,
        help="Name of index to reload. If 'all', all indices will be deleted.",
    )
    args = parser.parse_args()
    docs_directory = args.docs_dir
    index_name = args.index_name
    reload_index_name = args.reload
    print(f"Instantiating DocuBot for {docs_directory}")
    init()

    vector_store: Pinecone = None
    if reload_index_name:
        # Delete all pinecone indices. As a free user, we can only have one index at a time.
        delete_pinecone_index(reload_index_name)
        create_index(index_name)
        vector_store = create_vector_store(index_name, chunks=build_kb(docs_directory))
    elif index_name in pinecone.list_indexes():
        vector_store = fetch_vector_store(index_name)
    else:
        create_index(index_name)
        vector_store = create_vector_store(index_name, chunks=build_kb(docs_directory))
    chatbot(vector_store=vector_store)


if __name__ == "__main__":
    docubot()

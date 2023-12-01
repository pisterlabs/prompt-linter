#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This script allows you to ask questions of a directory of local documents. It
uses the GPT-3 model to create a vector index of the documents, and then allows
you to ask questions to the index.
'''

import os
import yaml
import openai
from llama_index import (
    GPTVectorStoreIndex,
    StorageContext,
    SimpleDirectoryReader,
    download_loader,
    load_index_from_storage
)
from llama_index.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores import SimpleVectorStore
from llama_index.storage.index_store import SimpleIndexStore
from argparse import ArgumentParser
from dotenv import load_dotenv

# script configuration
load_dotenv()

# We assume the OpenAI tokens are set in the environment as OPENAI_API_KEY
# and OPENAI_ORGANIZATION unless the environment variable OPENAI_CONFIG_PATH is
# set. If set, we read that yaml file and extract the access_token and
# organization_id from there.
def set_openai_credentials():
    openai_config_path = os.environ.get('OPENAI_CONFIG_PATH') or None

    if openai_config_path is not None:
        credentials = yaml.safe_load(open(opeai_config_path, "r"))

        os.environ["OPENAI_API_KEY"] = credentials["access_token"]
        os.environ["OPENAI_ORGANIZATION"] = credentials["organization_id"]

# Save the index in .JSON file for repeated use. Saves money on ADA API calls
def create_index_from_dir(data_dir, index_dir):
    # Load the documents from a directory.
    # We use SimpleDirectoryReader to read all the txt files in a folder
    # There are many options at https://llamahub.ai/
    documents = SimpleDirectoryReader(input_dir=data_dir, recursive=True).load_data()
    print(f"...loaded {len(documents)} documents from {data_dir}")

    # This example uses PDF reader, there are many options at https://llamahub.ai/
    # PDFReader = download_loader("PDFReader")
    # loader = PDFReader()
    # documents = loader.load_data(file=pdf_file)

    # Chunking and Embedding of the chunks.
    index = GPTVectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=index_dir)
    return index

def load_index(index_dir):
    storage_context = StorageContext.from_defaults(
        docstore=SimpleDocumentStore.from_persist_dir(persist_dir=index_dir),
        vector_store=SimpleVectorStore.from_persist_dir(persist_dir=index_dir),
        index_store=SimpleIndexStore.from_persist_dir(persist_dir=index_dir),
    )

    index = load_index_from_storage(storage_context)
    return index


def main(args):
    set_openai_credentials()

    user_data_dir = f"./data/{args.dir}/"
    user_index_dir = f"./indices/{args.dir}/"

    if args.create_index:
        print(f"Creating index from {user_data_dir} to {user_index_dir}...")
        index = create_index_from_dir(user_data_dir, user_index_dir)
        print("Index created.")
        return

    index = load_index(user_index_dir)

    # Retrieval, node poseprocessing, response synthesis. 
    query_engine = index.as_query_engine()

    question = ''
    print("\nAsk a question. Type 'quit' or ctrl-c to quit.\n")
    while True:
        print('> ', end='')
        try:
            question = input()
        except KeyboardInterrupt:
            print("\nExiting...")
            exit()

        if question == 'quit' or question == 'exit':
            break

        # Run the query engine on a user question.
        response = query_engine.query(question)
        print(f"\n{response}\n")

    print("Bye!\n")


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__, prog='rag.py', epilog='Have fun!')
    parser.add_argument('-c', '--create-index', help='(re)create the index', action='store_true')
    parser.add_argument('dir', help='data and indices subdirectory name')
    args = parser.parse_args()
    main(args)


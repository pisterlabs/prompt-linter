#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This script allows you to ask questions to the Alice in Wonderland book.
It uses the GPT-3 model to create a vector index of the book, and then
allows you to ask questions to the index.
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

# script configuration
persist_dir = "./indices/alice/"
pdf_file = "alice.pdf"

openai_config = "projects/infrastructure/charts/secrets/values/integration/openai-configuration/openai.yml"
credentials_path = os.path.join(os.path.expanduser('~'), openai_config)
credentials = yaml.safe_load(open(credentials_path, "r"))

os.environ["OPENAI_API_KEY"] = credentials["access_token"]
os.environ["OPENAI_ORGANIZATION"] = credentials["organization_id"]


# Save the index in .JSON file for repeated use. Saves money on ADA API calls
def create_index_from_pdf(persist_dir):
    # This example uses PDF reader, there are many options at https://llamahub.ai/
    # Use SimpleDirectoryReader to read all the txt files in a folder
    PDFReader = download_loader("PDFReader")
    loader = PDFReader()
    documents = loader.load_data(file=pdf_file)

    # Chunking and Embedding of the chunks.
    index = GPTVectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=persist_dir)
    return index

def load_index(persist_dir):
    storage_context = StorageContext.from_defaults(
        docstore=SimpleDocumentStore.from_persist_dir(persist_dir=persist_dir),
        vector_store=SimpleVectorStore.from_persist_dir(persist_dir=persist_dir),
        index_store=SimpleIndexStore.from_persist_dir(persist_dir=persist_dir),
    )

    index = load_index_from_storage(storage_context)
    return index


def main(args):
    print(args.question)

    if args.create_index:
        index = create_index_from_pdf(persist_dir)
    else:
        index = load_index(persist_dir)

    # Retrieval, node poseprocessing, response synthesis. 
    query_engine = index.as_query_engine()

    # Run the query engine on a user question.
    response = query_engine.query(args.question)
    print(response)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__, prog='alice.py', epilog='Have fun!')
    parser.add_argument('-c', '--create-index', help='(re)create the index', action='store_true')
    parser.add_argument('question', help='question string to ask the index')
    args = parser.parse_args()
    main(args)


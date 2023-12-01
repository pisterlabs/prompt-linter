# This is a simple script to embed a set of text files located in a data_directory, and store them in a Chroma collection.

import os
import argparse
import time
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from langchain.text_splitter import RecursiveCharacterTextSplitter
# load env
from dotenv import load_dotenv
load_dotenv()


def get_or_create_collection(collection_name: str, persist_directory: str):
    """
  Instantiates the Chroma client, and creates a collection, using OpenAI embeddings.
  """
    print(f"Creating collection {collection_name}")
    # Instantiate a persistent chroma client in the persist_directory.
    # Learn more at docs.trychroma.com
    client = chromadb.Client(settings=Settings(
        chroma_db_impl="duckdb+parquet", persist_directory=persist_directory))

    # We use the OpenAI embedding function.
    print("Using embedding function OpenAIEmbeddingFunction")
    embedding_function = OpenAIEmbeddingFunction(
        api_key=os.environ['OPENAI_API_KEY'])

    # If the collection already exists, we just return it. This allows us to add more
    # data to an existing collection.
    collection = client.get_or_create_collection(
        name=collection_name, embedding_function=embedding_function)

    print(f"Collection {collection_name} created.")
    return collection


def read_files_recursively(root_dir: str):
    print(f"Reading files from {root_dir}")
    documents = []
    metadatas = []
    r_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                chunk_overlap=200,
                                                separators=[" ", "\n"])
    print("Reading files recursively")
    for dirpath, dirnames, filenames in os.walk(root_dir):
        print(f"Reading directory {dirpath}")
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as file:
                    print(f"Reading file {filepath}")
                    chunks = r_splitter.split_text(file.read())
                    documents.extend(chunks)
                    metadatas.extend([{
                        'filename': filepath,
                        'chunk_id': i
                    } for i in range(len(chunks))])
            except UnicodeDecodeError:
                print(
                    f"Skipped file {filepath} due to Unicode decoding error.")
                
    print(f"Read {len(documents)} documents")
    return documents, metadatas


def main(data_dir: str, collection_name: str, persist_directory: str):
    documents, metadatas = read_files_recursively(data_dir)

    collection = get_or_create_collection(collection_name=collection_name,
                                          persist_directory=persist_directory)

    count = collection.count()
    print(f'Collection contains {count} documents')
    ids = [str(i) for i in range(count, count + len(documents))]
    # split documents into batches of 5k documents and add to collection with 1second delay
    batch_size = 1000
    for i in range(0, len(documents), batch_size):
        print(f'Adding batch {i}')
        batch_ids = ids[i:i+batch_size]
        batch_docs = documents[i:i+batch_size]
        batch_metas = metadatas[i:i+batch_size]
        collection.add(ids=batch_ids, documents=batch_docs, metadatas=batch_metas)
        time.sleep(1)

    collection.add(ids=ids, documents=documents, metadatas=metadatas)

    new_count = collection.count()
    print(f'Added {new_count - count} documents')


if __name__ == "__main__":
    # Read the data directory, collection name, and persist directory
    parser = argparse.ArgumentParser(
        description='Embed data from a directory into a Chroma collection')

    parser.add_argument(
        '--data_directory',
        type=str,
        help=
        'The directory where your text files are stored (or where you\'ll clone the repository)'
    )
    parser.add_argument(
        '--persist_directory',
        type=str,
        help='The directory where you want to store the Chroma collection')
    parser.add_argument('--collection_name',
                        type=str,
                        help='The name of the Chroma collection')

    # Parse arguments
    args = parser.parse_args()

    main(data_dir=args.data_directory,
         collection_name=args.collection_name,
         persist_directory=args.persist_directory)

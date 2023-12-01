# Langchain wraps the Milvus client and provides a few convenience methods for working with documents.
# It can split documents into chunks, embed them, and store them in Milvus.

import os
import argparse

from langchain.document_loaders import TextLoader, GitLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Milvus
from langchain.document_loaders import UnstructuredMarkdownLoader, DirectoryLoader
from langchain.document_loaders.text import TextLoader


from pymilvus import FieldSchema, DataType, CollectionSchema, Collection, connections, utility
from git import Repo


text_field = "text"
primary_field = "pk"
vector_field = "vector"


def load_documents(file_path, encoding='utf8', file_type='text'):
    if file_type == 'markdown':
        loader = UnstructuredMarkdownLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding=encoding)
    return loader.load()


def load_documents_from_github(github_url, file_type, collection_name):
    repoPath = "./docs/" + collection_name
    githubUrl = github_url
    # check if the repo exists
    if os.path.exists(repoPath):
        print("Repo already exists")
        githubUrl = None

    loader = GitLoader(
        clone_url=githubUrl,
        repo_path=repoPath,
        branch="main",
        file_filter=lambda file_path: file_path.endswith(
            ".{}".format(file_type)),
    )

    return loader.load()


def load_documents_from_directory(file_type, collection_name):
    path = "./docs/" + collection_name
    loader = DirectoryLoader(
        path=path,
        glob="**/*.{}".format(file_type),
        loader_cls=TextLoader
    )

    return loader.load()


def clone_from_github(github_url, collection_name):
    path = "./docs/" + collection_name
    if os.path.exists(path):
        print("Repo already exists")
        return

    repo = Repo.clone_from(
        github_url, to_path=path
    )
    repo.git.checkout("main")


def split_documents(documents, chunk_size=1000, chunk_overlap=0):
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)


def index_documents(milvus, docs):
    # Index the documents using the provided Milvus instance
    milvus.add_documents(docs)


def create_milvus_collection(embeddings, collection_name, host, port):
    # Connect to Milvus instance
    if not connections.has_connection("default"):
        connections.connect(host=host, port=port)
    utility.drop_collection(collection_name)
    # Create the collection in Milvus
    fields = []
    # Create the metadata field
    fields.append(
        FieldSchema('source', DataType.VARCHAR, max_length=200)
    )
    # Create the text field
    fields.append(
        FieldSchema(text_field, DataType.VARCHAR, max_length=2500)
    )
    # Create the primary key field
    fields.append(
        FieldSchema(primary_field, DataType.INT64,
                    is_primary=True, auto_id=True)
    )
    # Create the vector field
    fields.append(FieldSchema(vector_field, DataType.FLOAT_VECTOR, dim=1536))
    # Create the schema for the collection
    schema = CollectionSchema(fields)
    # Create the collection
    collection = Collection(collection_name, schema)
    # Index parameters for the collection
    index = {
        "index_type": "HNSW",
        "metric_type": "L2",
        "params": {"M": 8, "efConstruction": 64},
    }
    # Create the index
    collection.create_index(vector_field, index)

    # Create the VectorStore
    milvus = Milvus(
        embeddings,
        {"host": host, "port": port},
        collection_name,
        text_field,
    )

    return milvus


def main(input_dir, encoding, chunk_size, chunk_overlap, host, port, file_type, collection_name, github_url):
    embeddings = OpenAIEmbeddings()
    milvus = create_milvus_collection(embeddings, collection_name, host, port)

    documents = []
    if input_dir is not None:
        # Iterate through all the files in the input directory and process each one

        for file in os.listdir(input_dir):
            file_path = os.path.join(input_dir, file)
            if os.path.isfile(file_path):
                print(f"Processing {file_path}...")
                loadedDocuments = load_documents(
                    file_path, encoding, file_type)
                # concat two lists
                documents = documents + loadedDocuments
        docs = split_documents(documents, chunk_size, chunk_overlap)
        index_documents(milvus, docs)
        print(f"Indexed {len(docs)} chunks from {input_dir}.")

    if github_url is not None:
        documents = load_documents_from_directory(file_type, collection_name)
        docs = split_documents(documents, chunk_size, chunk_overlap)
        index_documents(milvus, docs)
        print(f"Indexed {len(docs)} chunks from {github_url}.")

    print("Done!")


# python src/index_documents.py --input_dir /path/to/your/documents --file_type markdown --collection_name my_collection
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Index documents for Question Answering over Documents application.")
    parser.add_argument('--input_dir', type=str, default=None,
                        help='Path to the directory containing documents to be indexed.')
    parser.add_argument('--encoding', type=str, default='utf8',
                        help='Encoding of the input documents.')
    parser.add_argument('--chunk_size', type=int, default=1000,
                        help='Size of the chunks to split documents into.')
    parser.add_argument('--chunk_overlap', type=int, default=0,
                        help='Number of overlapping characters between consecutive chunks.')
    parser.add_argument('--host', type=str, default="127.0.0.1",
                        help='Host address for the Milvus server.')
    parser.add_argument('--port', type=str, default="19530",
                        help='Port for the Milvus server.')
    parser.add_argument('--file_type', type=str, default="text", choices=[
                        "text", "markdown", "adoc"], help='Type of the input files (text or markdown).')
    parser.add_argument('--collection_name', type=str, required=True,
                        help='Name of the collection to index the documents into.')
    parser.add_argument('--github_url', type=str, default=None,
                        help='URL of the file to download from GitHub (raw content URL).')

    args = parser.parse_args()

    main(args.input_dir, args.encoding, args.chunk_size, args.chunk_overlap,
         args.host, args.port, args.file_type, args.collection_name, args.github_url)

""" Gets MD5 Hash of file. """
import hashlib
import os
import traceback

from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import milvus

from lib.environment_variables import ENV

default_ignores = [
    "*.git*",
    "docker/",
    ".github/",
    "tox.ini",
    "*.db",
    ".pyc"
]


def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest().encode()


def chunk_docs(docs):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunked_docs = text_splitter.split_documents(docs)
    print(f"{len(chunked_docs)}")
    return chunked_docs


def get_ignore_patterns(ignore_path, additional_patterns=None):
    patterns = default_ignores
    if additional_patterns is not None:
        patterns += additional_patterns
    gitignore_patterns = read_gitignore(ignore_path) if os.path.isfile(ignore_path) else []
    return gitignore_patterns + patterns


def read_gitignore(gitignore_path):
    if not os.path.isfile(gitignore_path):
        return []
    with open(gitignore_path, 'r') as f: content = f.readlines()
    return content or []


def create_embeddings(docs):
    try:
        embeddings = OpenAIEmbeddings(disallowed_special=())
        vector_db = milvus.Milvus.from_documents(
            docs,
            embeddings,
            connection_args={"host": ENV["MILVUS_HOST"], "port": ENV["MILVUS_PORT"]},
        )
    except Exception as e:
        print(e)
        traceback.print_exc()

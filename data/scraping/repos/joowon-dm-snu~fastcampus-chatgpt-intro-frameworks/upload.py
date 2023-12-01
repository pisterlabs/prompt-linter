import os

from dotenv import load_dotenv
from langchain.document_loaders import (
    NotebookLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

load_dotenv()

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(CUR_DIR), "dataset")

SK_CODE_DIR = os.path.join(DATA_DIR, "semantic-kernel", "python")
SK_SAMPLE_DIR = os.path.join(
    DATA_DIR, "semantic-kernel", "samples", "notebooks", "python"
)
SK_DOC_DIR = os.path.join(DATA_DIR, "semantic-kernel-docs", "semantic-kernel")

CHROMA_PERSIST_DIR = os.path.join(CUR_DIR, "chroma-persist")
CHROMA_COLLECTION_NAME = "fastcampus-bot"


LOADER_DICT = {
    "py": TextLoader,
    "md": UnstructuredMarkdownLoader,
    "ipynb": NotebookLoader,
}


def upload_embedding_from_file(file_path):
    loader = LOADER_DICT.get(file_path.split(".")[-1])
    if loader is None:
        raise ValueError("Not supported file type")
    documents = loader(file_path).load()

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    Chroma.from_documents(
        docs,
        OpenAIEmbeddings(),
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
    )


def upload_embeddings_from_dir(dir_path):
    failed_upload_files = []

    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith(".py") or file.endswith(".md") or file.endswith(".ipynb"):
                file_path = os.path.join(root, file)

                try:
                    upload_embedding_from_file(file_path)
                    print("SUCCESS: ", file_path)
                except Exception:
                    print("FAILED: ", file_path)
                    failed_upload_files.append(file_path)


if __name__ == "__main__":
    upload_embeddings_from_dir(SK_CODE_DIR)
    upload_embeddings_from_dir(SK_SAMPLE_DIR)
    upload_embeddings_from_dir(SK_DOC_DIR)

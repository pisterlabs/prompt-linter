
#JMRingest June 19 Updated with ability to choose or create folders for persist directory, and source directory.
#Process documents of many types from source_directory="/Users/jonathanrothberg/Desktop/SourceForVectors"
#Write them to VectorStore
#Added print statements do you know which files are being processed.
#Made local directory for embeddings
import os
import glob
from typing import List
import time
import tkinter as tk
from tkinter import filedialog
from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from chromadb.config import Settings


embeddings_model_name = "/Users/jonathanrothberg/all-MiniLM-L6-v2" #added this locally so not in .cache


def select_directories():
    root = tk.Tk()
    root.withdraw()

    # Select source directory
    print("Select the source directory:")
    source_directory = filedialog.askdirectory(initialdir=os.path.expanduser("~"))

    # Select persist directory
    print("Select the persist directory:")
    persist_directory = filedialog.askdirectory(initialdir=os.path.expanduser("~"))

    # Create the persist directory if it doesn't exist
    os.makedirs(persist_directory, exist_ok=True)

    print("Source directory:", source_directory)
    print("Persist directory:", persist_directory)

    return source_directory, persist_directory


class MyElmLoader(UnstructuredEmailLoader):
    """Wrapper to fallback to text/plain when default does not work"""

    def load(self) -> List[Document]:
        """Wrapper adding fallback for elm without html"""
        try:
            try:
                doc = UnstructuredEmailLoader.load(self)
            except ValueError as e:
                if 'text/html content not found in email' in str(e):
                    # Try plain text
                    self.unstructured_kwargs["content_source"]="text/plain"
                    doc = UnstructuredEmailLoader.load(self)
                else:
                    raise
        except Exception as e:
            # Add file_path to exception message
            raise type(e)(f"{self.file_path}: {e}") from e

        return doc


def load_single_document(file_path: str) -> Document: #entire path in source name
    ext = "." + file_path.rsplit(".", 1)[-1]
    #print ("file_path: ", file_path)
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()[0]

    raise ValueError(f"Unsupported file extension '{ext}'")


def load_documents(source_dir: str) -> List[Document]:
    # Loads all documents from source documents directory
    all_files = []
    print ("source_dir: ", source_dir)
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )
    return [load_single_document(file_path) for file_path in all_files]


def main():
    # Load documents and split in chunks
    source_directory, persist_directory = select_directories()
    chunk_size = 500
    chunk_overlap = 50
    print(f"Loading documents from {source_directory}")
    documents = load_documents(source_directory)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    print(f"Loaded {len(documents)} documents from {source_directory}")
    print(f"Split into {len(texts)} chunks of text (max. {chunk_size} characters each)")

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    CHROMA_SETTINGS = Settings(
        chroma_db_impl='duckdb+parquet',
        persist_directory=persist_directory,
        anonymized_telemetry=False)
    # Create and store locally vectorstore
    print ("Start time: "+time.strftime("%m%d-%H%M"))
    db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)
    db.persist()
    db = None
    print ("Finish time: "+time.strftime("%m%d-%H%M"))
    print ("done")

# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    # ".docx": (Docx2txtLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".eml": (MyElmLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PDFMinerLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    # Add more mappings for other file extensions and loaders as needed
}


if __name__ == "__main__":
    main()

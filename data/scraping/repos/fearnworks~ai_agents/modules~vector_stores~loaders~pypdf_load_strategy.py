from typing import List, Iterable
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from loguru import logger
from langchain.text_splitter import RecursiveCharacterTextSplitter
from modules.vector_stores.embedding_bases import DocumentLoadStrategy
from langchain.schema import Document
from dataclasses import dataclass

@dataclass
class PyPDFConfig:
    dir_location: str
    glob_pattern: str = "./*.pdf"
    chunk_size: int = 1000
    chunk_overlap: int =    200

class PyPDFLoadStrategy(DocumentLoadStrategy):
    def __init__(self, config: PyPDFConfig):
        """
        A document load strategy that loads PDF files using PyPDF.

        Args:
            dir_path (str): The directory path to load PDF files from.
            glob_pattern (str): The glob pattern to match PDF files.

        Attributes:
            logger (logging.Logger): The logger instance for this class.
            dir_path (str): The directory path to load PDF files from.
            glob_pattern (str): The glob pattern to match PDF files.
        """
        self.logger = logger
        self.dir_path = config.dir_location
        self.glob_pattern = config.glob_pattern
        self.chunk_size = config.chunk_size
        self.chunk_overlap = config.chunk_overlap


    def load(self) -> Iterable[Document]:
        """
        Loads PDF files from the specified directory path and returns an iterable of `Document` instances.

        Returns:
            Iterable[Document]: An iterable of `Document` instances.
        """
        loader = DirectoryLoader(
            self.dir_path, glob=self.glob_pattern, loader_cls=PyPDFLoader
        )  # Note: If you're using PyPDFLoader then it will split by page for you already
        documents = loader.load()
        self.logger.info(f"Loaded {len(documents)} documents from {self.dir_path}")
        return documents

    def split(self, documents: Iterable[Document]):
        """
        Splits the specified list of PyPDFLoader instances into text chunks using a recursive character text splitter.

        Args:
            documents  (Iterable[Document]): The documents to split.
            chunk_size (int): The size of each text chunk.
            chunk_overlap (int): The overlap between adjacent text chunks.

        Returns:
            List[str]: A list of text chunks.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        texts = text_splitter.split_documents(documents)
        self.logger.info(f"Split {len(documents)} documents into {len(texts)}")
        return texts


def get_default_pypdf_loader(dir_location: str) -> PyPDFLoadStrategy:
    dir_path = dir_location

    config: PyPDFConfig = PyPDFConfig(
        dir_location=dir_path
    )
    return PyPDFLoadStrategy(config)

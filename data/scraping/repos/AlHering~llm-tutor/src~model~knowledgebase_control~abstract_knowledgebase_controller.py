# -*- coding: utf-8 -*-
"""
****************************************************
*                    LLM Tutor                     *
*            (c) 2023 Alexander Hering             *
****************************************************
"""
import os
import abc
from typing import Any, List, Tuple
from chromadb.api.types import EmbeddingFunction, Embeddings, Documents
from langchain.docstore.document import Document
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
from src.utility.bronze import langchain_utility


def reload_document(document_path: str) -> Document:
    """
    Function for (re)loading document content.
    :param document_path: Document path.
    :return: Document object.
    """
    res = langchain_utility.DOCUMENT_LOADERS[os.path.splitext(document_path)[
        1]](document_path).load()
    return res[0] if isinstance(res, list) and len(res) == 1 else res


class KnowledgeBaseController(abc.ABC):
    """
    Abdstract class for knowledge base controllers.
    """

    @abc.abstractmethod
    def get_or_create_collection(self, name: str, metadata: dict = None, embedding_function: EmbeddingFunction = None) -> Any:
        """
        Method for retrieving or creating a collection.
        :param name: Collection name.
        :param metadata: Embedding collection metadata. Defaults to None.
        :param embedding_function: Embedding function for the collection. Defaults to base embedding function.
        :return: Database API.
        """
        pass

    @abc.abstractmethod
    def get_retriever(self, name: str, search_type: str = "similarity", search_kwargs: dict = {"k": 4, "include_metadata": True}) -> VectorStoreRetriever:
        """
        Method for acquiring a retriever.
        :param name: Collection to use.
        :param search_type: The retriever's search type. Defaults to "similarity".
        :param search_kwargs: The retrievery search keyword arguments. Defaults to {"k": 4, "include_metadata": True}.
        :return: Retriever instance.
        """
        pass

    @abc.abstractmethod
    def embed_documents(self, name: str, documents: List[Document], ids: List[str] = None) -> None:
        """
        Method for embedding documents.
        :param name: Collection to use.
        :param documents: Documents to embed.
        :param ids: Custom IDs to add. Defaults to the hash of the document contents.
        """
        pass

    def load_folder(self, folder: str, target_collection: str = "base", splitting: Tuple[int] = None) -> None:
        """
        Method for (re)loading folder contents.
        :param folder: Folder path.
        :param target_collection: Collection to handle folder contents. Defaults to "base".
        :param splitting: A tuple of chunk size and overlap for splitting. Defaults to None in which case the documents are not split.
        """
        file_paths = []
        for root, dirs, files in os.walk(folder, topdown=True):
            file_paths.extend([os.path.join(root, file) for file in files])

        self.load_files(file_paths, target_collection, splitting)

    def load_files(self, file_paths: List[str], target_collection: str = "base", splitting: Tuple[int] = None) -> None:
        """
        Method for (re)loading file paths.
        :param file_paths: List of file paths.
        :param target_collection: Collection to handle folder contents. Defaults to "base".
        :param splitting: A tuple of chunk size and overlap for splitting. Defaults to None in which case the documents are not split.
        """
        document_paths = [file for file in file_paths if any(file.lower().endswith(
            supported_extension) for supported_extension in langchain_utility.DOCUMENT_LOADERS)]
        documents = []

        with tqdm(total=len(document_paths), desc="(Re)loading folder contents...", ncols=80) as progress_bar:
            for index, document_path in enumerate(document_paths):
                documents.append(reload_document(document_path))
                progress_bar.update(index)

        if splitting is not None:
            documents = self.split_documents(documents, *splitting)

        self.embed_documents(target_collection, documents)

    def split_documents(self, documents: List[Document], split: int, overlap: int) -> List[Document]:
        """
        Method for splitting document content.
        :param documents: Documents to split.
        :param split: Chunk size to split documents into.
        :param overlap: Overlap for split chunks.
        :return: Split documents.
        """
        return RecursiveCharacterTextSplitter(
            chunk_size=split,
            chunk_overlap=overlap,
            length_function=len).split_documents(documents)

"""Loading logic for loading documents from an AWS S3 file."""
import os
import tempfile
from abc import ABCMeta, ABC
from typing import List, Dict, Type, Iterator
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders.unstructured import UnstructuredFileLoader
from src.api import S3Connector
from src.service.chatbot.loaders.file_loader_factory import FileLoaderFactory
from utilities import get_file_extension


class S3FileLoader(BaseLoader):

    def __init__(self, object_id: str):
        """
        """
        self.object_id = object_id
        self.s3_connector = S3Connector()

    def load(self) -> List[Document]:
        """Load documents."""
        s3connector = self.s3_connector
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = f"{temp_dir}/{self.object_id}"
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            s3connector.download_object(object_id=self.object_id, file_path=file_path)
            loader = FileLoaderFactory.get_loader(file_path)
            return loader.load()

    def lazy_load(self) -> Iterator[Document]:
        # TODO: Implement lazy loading
        pass





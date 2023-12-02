"""Loading logic for loading documents from an AWS S3 directory."""
from typing import List, Iterator
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

from src.api import S3Connector
from src.service.chatbot.loaders.s3_file_loader import S3FileLoader


class S3DirectoryLoader(BaseLoader):
    """Loading logic for loading documents from an AWS S3."""

    def __init__(self, prefix: str = ""):
        """Initialize with bucket and key name.

        Args:
            bucket: The name of the S3 bucket.
            prefix: The prefix of the S3 key. Defaults to "".
        """
        self.prefix = prefix
        self.s3_connector = S3Connector()

    def load(self) -> List[Document]:
        """Load documents."""

        s3connector = self.s3_connector
        objects_ids = s3connector.get_object_ids(prefix=self.prefix)
        docs = []
        for object_id in objects_ids:
            loader = S3FileLoader(object_id=object_id)
            docs.extend(loader.load())
        return docs

    def lazy_load(self) -> Iterator[Document]:
        # TODO: Implement lazy loading
        pass

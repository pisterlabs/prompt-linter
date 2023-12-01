from simpletuning import DataRetriver
from langchain.document_loaders.directory import DirectoryLoader
from langchain.document_loaders.base import Document

from typing import List
from pathlib import Path


class LcDirectoryRetriver(DataRetriver):
    """
    This class is a wrapper for string data, just returning the input as it is.
    """

    path: Path

    def __init__(self, path: str) -> None:
        self.path = Path(path)

    def retrive(self) -> List[Document]:
        return DirectoryLoader(self.path.as_posix()).load()

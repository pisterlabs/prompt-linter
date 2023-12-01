# coding=utf-8

from .base_loader import BaseFileLoader
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders.base import BaseLoader

class TextLoader(BaseFileLoader):
    def __init__(self, text_folder_path):
        super().__init__(text_folder_path)

    def build_loader(self) -> BaseLoader:
        self.loader = DirectoryLoader(self.folder_path, glob="**/*.txt") 
        return self.loader

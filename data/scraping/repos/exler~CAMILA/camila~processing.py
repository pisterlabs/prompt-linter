from abc import abstractmethod
from typing import Self

from langchain.docstore.document import Document
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


class Processor:
    def __init__(self: Self, documents: list[Document]) -> None:
        self._docs = documents

    @abstractmethod
    def process(self: Self) -> list[Document]:
        pass


class MarkdownProcessor(Processor):
    def process(self: Self) -> list[Document]:
        md_header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("##", "Section"), ("###", "Subsection")])
        md_header_splits = []
        for doc in self._docs:
            md_header_splits.extend(md_header_splitter.split_text(doc.page_content))

        return md_header_splits


class CharacterProcessor(Processor):
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 0

    def process(self: Self) -> list[Document]:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.CHUNK_SIZE, chunk_overlap=self.CHUNK_OVERLAP)
        all_splits = text_splitter.split_documents(self._docs)

        return all_splits

"""Split on markdown headers first, then recursively."""

import re
from typing import Any
from typing import Callable
from typing import Optional
from typing import Sequence
from typing import Sized

from langchain.schema.document import BaseDocumentTransformer
from langchain.schema.document import Document
from langchain.text_splitter import Language
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm


def _is_valid_section(doc: Document) -> bool:
    """Check if a section is valid. Currently a section is valid if it contains at least one letter."""
    return bool(re.search(r"[a-zA-Z]", doc.page_content))


class RecursiveMarkdownTextSplitter(BaseDocumentTransformer):
    """Split documents using upon a trained model."""

    def __init__(
        self,
        headers_to_split_on: Optional[list[tuple[str, str]]] = None,
        title_header_separator: str = " ",
        chunk_size: int = 2000,
        chunk_overlap: int = 200,
        length_function: Callable[[Sized], int] = len,
        **kwargs: Any
    ):
        """Initialize markdown header and recursive character text splitters."""
        super(BaseDocumentTransformer, self).__init__()

        if headers_to_split_on is None:
            headers_to_split_on = [("##", "Header 2")]
        self.headers_to_split_on = headers_to_split_on
        self.title_header_separator = title_header_separator
        self.markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        self.text_splitter = RecursiveCharacterTextSplitter.from_language(
            Language.MARKDOWN,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
        )

    def transform_documents(self, documents: Sequence[Document], **kwargs: Any) -> Sequence[Document]:
        """Transform documents by splitting them first using the markdown header splitter, then recursive."""
        verbose = kwargs.get("verbose", False)
        splits: list[Document] = []

        # first, split on markdown headers
        for doc in tqdm(documents, disable=not verbose):
            sections = self.markdown_splitter.split_text(doc.page_content)
            for section in sections:
                if not _is_valid_section(section):
                    continue
                metadata = doc.metadata.copy()
                # append headers to title?
                if self.title_header_separator:
                    for _, header in self.headers_to_split_on:
                        if header in section.metadata:
                            metadata["title"] += self.title_header_separator + section.metadata[header]
                splits.append(Document(metadata=metadata, page_content=section.page_content))

        # next, split recursively on characters
        splits = self.text_splitter.split_documents(splits)

        return splits

    async def atransform_documents(self, documents: Sequence[Document], **kwargs: Any) -> Sequence[Document]:
        """Transform documents asynchronously."""
        raise NotImplementedError

    def split_documents(self, documents: Sequence[Document], verbose: bool = False) -> list[Document]:
        """Split documents using model."""
        return list(self.transform_documents(documents, verbose=verbose))

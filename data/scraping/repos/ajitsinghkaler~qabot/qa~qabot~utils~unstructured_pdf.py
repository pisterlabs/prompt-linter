from typing import List
from langchain.document_loaders.unstructured import UnstructuredFileLoader, UnstructuredFileIOLoader

class UnstructuredPDFLoader2(UnstructuredFileIOLoader):
    """Loader that uses unstructured to load PDF files."""

    def _get_elements(self) -> List:
        from unstructured.partition.pdf import partition_pdf
        return partition_pdf(file=self.file)

    def _get_metadata(self) -> dict:
        return {**self.unstructured_kwargs}
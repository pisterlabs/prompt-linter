from langchain.document_loaders import UnstructuredFileLoader
from langchain.docstore.document import Document

from typing import List


def load_pptx_text(filePath: str) -> List[Document]:
    """This loader loads PowerPoint presentations one page at a time. It does
    its best to caputre the text in each PowerPoint slide and stores each slide
    as one Document (langchain.docstore.document.Document).
    """
    return UnstructuredFileLoader(
        file_path=filePath,
        mode="paged",  # "single", "elements", "paged"
        unstructured_kwargs=None
    ).load()


def load_pptx(filePath):
    return load_pptx_text(filePath)

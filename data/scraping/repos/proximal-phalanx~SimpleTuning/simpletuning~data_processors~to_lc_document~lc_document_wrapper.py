from typing import Any, List
from langchain.vectorstores.base import Document
from simpletuning.data_processor import DataProcessor


class LcDocumentWrapper(DataProcessor):
    """
    This transformer is used to transform the data into list of langchain documents.
    """

    def valid(self, data: Any) -> bool:
        if isinstance(data, Document):
            return True
        elif isinstance(data, list) and all(isinstance(x, Document) for x in data):
            return True
        elif isinstance(data, str):
            return True
        return False

    def transform(self, data: Any) -> List[Document] | Document:
        if isinstance(data, Document):
            return data
        elif isinstance(data, list) and all(isinstance(x, Document) for x in data):
            return data
        elif isinstance(data, str):
            return Document(page_content=data, metadata={"source": "string"})
        raise Exception(
            f"This transformer is not valid for the data type {type(data)}."
        )

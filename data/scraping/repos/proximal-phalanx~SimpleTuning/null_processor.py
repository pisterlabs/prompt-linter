from typing import Any, List
from langchain.vectorstores.base import Document
from simpletuning.data_processor import DataProcessor


class NullProcessor(DataProcessor):
    """
    This transformer returns the data.
    If the data is not in format of List, then return [data].
    """

    def valid(self, data: Any) -> bool:
        return True

    def transform(self, data: Any) -> List[Document]:
        return data if isinstance(data, list) else [data]

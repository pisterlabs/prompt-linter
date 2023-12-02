"""Loader that loads local any json files."""
import json
import json.decoder
from typing import Any, List

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


def _stringify_value(val: Any) -> str:
    if isinstance(val, str):
        return val
    elif isinstance(val, dict):
        return "\n" + _stringify_dict(val)
    elif isinstance(val, list):
        return "\n".join(_stringify_value(v) for v in val)
    else:
        return str(val)


def _stringify_dict(data: dict) -> str:
    text = ""
    for key, value in data.items():
        text += key + ": " + _stringify_value(data[key]) + "\n"
    return text


class JSONLoader(BaseLoader):
    """Loader that loads json files."""

    def __init__(self, file_path: str):
        """Initialize with file path. """
        self.file_path = file_path


    def load(self) -> List[Document]:
        """Load file."""
        text = ""
        with open(self.file_path, "r") as file:  # Use a context manager
            """Load a JSON file and return its content as a dictionary."""
            data = json.load(file)
            text += _stringify_dict(data)
            metadata = {"source": self.file_path}
            return [Document(page_content=text, metadata=metadata)]
import json
from os import path
from typing import List
from parsers.base import BaseParser
from parsers.pdf import PdfParser
import langchain.docstore.document as docstore


class JsonParser(BaseParser):
    """A parser for extracting and cleaning text from PDF documents."""

    def __init__(self, **kwargs):
        self.vortex_pdf_parser = PdfParser(**kwargs)
        super().__init__(**kwargs)

    def text_to_docs(self, file: str, path_root="./") -> List[docstore.Document]:
        """Split the text into chunks and return them as Documents."""
        with open(file, "r") as doc:
            data = json.load(doc)
        metadata = data.get("metadata")
        match metadata.get("type"):
            case "pdf":
                file_path = metadata.get("path")
                full_path = path.join(path_root, file_path)
                return self.vortex_pdf_parser.text_to_docs(full_path, metadata)
            case "json":
                return self.load_pure_json(data, metadata)

    def load_pure_json(self, data: dict, metadata: dict):
        text = self.clean_text([data.get("content")])
        return self._docs_builder(text, data.get("metadata"))

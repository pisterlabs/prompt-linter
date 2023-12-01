""" Temporary class created for windows, since jq doesnt work, will load a document usable by langchain"""
import json
from pathlib import Path
from typing import List, Optional, Union

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class JSONLoader(BaseLoader):
    """Custom JSON loader for loading wikipedia data into langchain"""

    def __init__(
        self,
        file_path: Union[str, Path],
        content_key: Optional[str] = None,
        fulltext: bool = False,
    ):
        self.file_path = Path(file_path).resolve()
        self._content_key = content_key
        self.fulltext = fulltext

    def load(self) -> List[Document]:
        """Load and return documents from the JSON file."""

        docs = []
        # Load JSON file
        with open(self.file_path) as file:
            data = json.load(file)

            # Iterate through 'pages'
        for government_name, government_text in data.items():
            base_metadata = {"gov_type": government_name}

            summary = government_text["summary"]
            summary_metadata = base_metadata.copy()
            summary_metadata["type"] = "summary"
            docs.append(Document(page_content=summary, metadata=summary_metadata))
            if self.fulltext:
                full_text = government_text["full_page_content"]
                full_text_metadata = base_metadata.copy()
                full_text_metadata["type"] = "fulltext"
                docs.append(
                    Document(page_content=full_text, metadata=full_text_metadata)
                )

        return docs

"""Loader that loads Scrapbox exported json file."""
from pathlib import Path
from typing import List
import json

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class ScrapboxLoader(BaseLoader):
    """Loader that loads Roam files from disk."""

    def __init__(self, path: str):
        """Initialize with path."""
        self.file_path = path

    def load(self) -> List[Document]:
        """Load documents."""
        # Load the JSON file into a dictionary
        with open(self.file_path, 'r') as file:
            data = json.load(file)
        # Loop through the pages and extract the text from each page's lines array
        pageTexts = {}
        for page in data['pages']:
            for line in page['lines']:
                if isinstance(line, dict) and 'text' in line:
                    # If the line is a dictionary with a 'text' key, extract the text
                    if page['title'] in pageTexts:
                        pageTexts[page['title']] += line['text']
                    else:
                        pageTexts[page['title']] = line['text']
                elif isinstance(line, str):
                    # If the line is a string, it's assumed to be the text itself
                    if page['title'] in pageTexts:
                        pageTexts[page['title']] += line
                    else:
                        pageTexts[page['title']] = line
                else:
                    # Otherwise, the line doesn't have any text to extract
                    continue

        docs = []
        for title, pageText in pageTexts.items():
            metadata = {"source": title}
            docs.append(Document(page_content=pageText, metadata=metadata))
        return docs
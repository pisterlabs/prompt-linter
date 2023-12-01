"""Load Church Magazines."""

import json
import os
from typing import Any
from typing import Iterator

from bs4 import BeautifulSoup
from langchain.document_loaders.base import BaseLoader
from langchain.schema.document import Document
from tqdm import tqdm

from models.load_utils import clean
from models.load_utils import to_markdown


def extract_title(soup: BeautifulSoup) -> Any:
    """Extract the title from the page."""
    # get the title
    title = soup.select_one("article header h1")
    return title


def extract_content(soup: BeautifulSoup) -> Any:
    """Extract the HTML content from the page."""
    # Find all sections
    content = soup.find("div", class_="body-block")
    return content


def load_magazine(url: str, html: str, bs_parser: str = "html.parser") -> Document:
    """Load Church Magazines from a url and html."""
    soup = BeautifulSoup(html, "html.parser")
    title = extract_title(soup)
    content = extract_content(soup)

    content = clean(to_markdown(str(content), base_url=url)) if content else ""

    metadata = {
        "url": url,
        "title": clean(title) if title else "",
    }
    print(metadata)
    return Document(page_content=content, metadata=metadata)


class MagazineLoader(BaseLoader):
    """Loader for Church Magazines."""

    def lazy_load(self) -> Iterator[Document]:
        """A lazy loader for Documents."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement lazy_load()")

    def __init__(self, path: str = "", bs_parser: str = "html.parser"):
        """Initialize loader."""
        super().__init__()
        self.path = path
        self.bs_parser = bs_parser

    def load(self, verbose: bool = False) -> list[Document]:
        """Load documents from path."""
        docs = []
        for filename in tqdm(os.listdir(self.path), disable=not verbose):
            path = os.path.join(self.path, filename)
            with open(path, encoding="utf8") as f:
                data = json.load(f)
            doc = load_magazine(data["url"], data["html"], bs_parser=self.bs_parser)
            if not doc.metadata["title"] or not doc.page_content:
                if verbose:
                    print("Missing title or content - skipping", filename)
                continue

            docs.append(doc)
        return docs

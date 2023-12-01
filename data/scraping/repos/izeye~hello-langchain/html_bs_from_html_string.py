"""Loader that uses bs4 to load HTML strings, enriching metadata with page title."""

import logging
from typing import Dict, List, Union

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

logger = logging.getLogger(__name__)


class BSHTMLLoader(BaseLoader):
    """Loader that uses beautiful soup to parse HTML strings."""

    def __init__(
        self,
        source: str,
        html_string: str,
        bs_kwargs: Union[dict, None] = None,
        get_text_separator: str = "",
    ) -> None:
        """Initialise with source, HTML string, and optionally, file encoding to use, and any kwargs
        to pass to the BeautifulSoup object."""
        try:
            import bs4  # noqa:F401
        except ImportError:
            raise ValueError(
                "beautifulsoup4 package not found, please install it with "
                "`pip install beautifulsoup4`"
            )

        self.source = source
        self.html_string = html_string
        if bs_kwargs is None:
            bs_kwargs = {"features": "lxml"}
        self.bs_kwargs = bs_kwargs
        self.get_text_separator = get_text_separator

    def load(self) -> List[Document]:
        from bs4 import BeautifulSoup

        """Load HTML document into document objects."""
        soup = BeautifulSoup(self.html_string, **self.bs_kwargs)

        text = soup.get_text(self.get_text_separator)

        if soup.title:
            title = str(soup.title.string)
        else:
            title = ""

        metadata: Dict[str, Union[str, None]] = {
            "source": self.source,
            "title": title,
        }
        return [Document(page_content=text, metadata=metadata)]

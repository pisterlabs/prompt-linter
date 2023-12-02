# -*- coding: utf-8 -*-


from typing import Dict, List, Union
from langchain.docstore.document import Document
from langchain.doucment_loaders.base import BaseLoader


class HTMLLoader(BaseLoader):
    """Loader that uses beautiful soup to parse HTML files."""

    def __init__(
        self,
        file_path: str,
        is_online: bool = False,
        open_encoding: Union[str, None] = None,
        bs_kwargs: Union[dict, None] = None,
        get_text_separator: str = "",
    ) -> None:
        """Initialise with path, and optionally, file encoding to use, and any kwargs
        to pass to the BeautifulSoup object."""
        try:
            import bs4  # noqa:F401
        except ImportError:
            raise ValueError(
                "beautifulsoup4 package not found, please install it with "
                "`pip install beautifulsoup4`"
            )

        self.file_path = file_path
        self.is_online = is_online
        self.open_encoding = open_encoding
        if bs_kwargs is None:
            bs_kwargs = {"features": "lxml"}
        self.bs_kwargs = bs_kwargs
        self.get_text_separator = get_text_separator

    def load(self) -> List[Document]:
        from bs4 import BeautifulSoup

        if not self.is_online:
            """Load HTML document into document objects."""
            with open(self.file_path, "r", encoding=self.open_encoding) as f:
                soup = BeautifulSoup(f, **self.bs_kwargs)
        else:
            import requests
            response = requests.get(self.file_path)
            html = response.content
            soup = BeautifulSoup(html, **self.bs_kwargs)

        text = self.__raw_content_parser(soup)
        if soup.title:
            title = str(soup.title.string)
        else:
            title = ""
        metadata: Dict[str, Union[str, None]] = {
            "source": self.file_path,
            "title": title,
        }
        return [Document(page_content=text, metadata=metadata)]
    
    def __raw_content_parser(self, soup):
        return soup.get_text(self.get_text_separator)
    
    def __content_parser(self, soup):
        # TODO
        pass
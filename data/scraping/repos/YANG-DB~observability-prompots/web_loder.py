"""Web base loader class."""
import logging
from typing import Any,Set, List, Optional

import requests

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

logger = logging.getLogger(__file__)

default_header_template = {
    "User-Agent": "",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*"
              ";q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.google.com/",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}


import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

logger = logging.getLogger(__file__)

default_header_template = {
    "User-Agent": "",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*"
              ";q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.google.com/",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}


class WebBaseLoader(BaseLoader):
    """Loader that uses urllib and beautiful soup to load webpages."""

    def __init__(self, web_path: str, header_template: Optional[dict] = None, depth: int = 3):
        """Initialize with webpage path."""
        self.web_path = web_path
        self.depth = depth
        self.session = requests.Session()

        try:
            from fake_useragent import UserAgent

            headers = header_template or default_header_template
            headers["User-Agent"] = UserAgent().random
            self.session.headers = dict(headers)
        except ImportError:
            logger.info(
                "fake_useragent not found, using default user agent."
                "To get a realistic header for requests, `pip install fake_useragent`."
            )

    def _scrape(self, url: str) -> Any:
        html_doc = self.session.get(url)
        soup = BeautifulSoup(html_doc.text, "html.parser")
        print(f"--- Scraping {url}")
        return soup

    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> Set[str]:
        links = set()
        for link in soup.find_all("a"):
            href = link.get("href")
            if href:
                full_url = urljoin(base_url, href)
                links.add(full_url)
        return links

    def _crawl(self, url: str, depth: int, visited: Set[str]) -> List[str]:
        if depth == 0 or url in visited:
            return []

        visited.add(url)
        soup = self._scrape(url)
        links = self._extract_links(soup, url)
        all_links = [url]

        for link in links:
            all_links += self._crawl(link, depth - 1, visited)

        return all_links

    def load(self) -> List[Document]:
        visited = set()
        urls = self._crawl(self.web_path, self.depth, visited)

        documents = []
        for url in urls:
            soup = self._scrape(url)
            text = soup.get_text()
            metadata = {"source": url}
            documents.append(Document(page_content=text, metadata=metadata))

        return documents
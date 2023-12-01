
import itertools
import re
from typing import Any, Callable, Generator, Iterable, List, Optional
from langchain.document_loaders.sitemap import SitemapLoader, _batch_block
from langchain.document_loaders.web_base import WebBaseLoader
from langchain.schema import Document
import asyncio
from typing import Any, Dict, List, Optional, Union
import random
import math

class CustomSitemapLoader(SitemapLoader):
    
    def __init__(self, web_path, scrapping_factor: float):
         self.scrapping_factor = scrapping_factor
         super().__init__(web_path)

    def parse_sitemap(self, soup: Any, mode) -> List[dict]:
        """Parse sitemap xml and load into a list of dicts."""
        els = []
        for url in soup.find_all("url"):
            loc = url.find("loc")
            if not loc:
                continue

            # Strip leading and trailing whitespace and newlines
            loc_text = loc.text.strip()

            if self.filter_urls and not any(
                re.match(r, loc_text) for r in self.filter_urls
            ):
                continue

            els.append(
                {
                    tag: prop.text
                    for tag in ["loc", "lastmod", "changefreq", "priority"]
                    if (prop := url.find(tag))
                }
            )

        for sitemap in soup.find_all("sitemap"):
            loc = sitemap.find("loc")
            if not loc:
                continue
            soup_child = self.scrape_all([loc.text], "xml", mode)[0]
            els.extend(self.parse_sitemap(soup_child, mode))

        return els

    def scrape_all(self, urls: List[str], parser: Union[str, None] = None, mode = 'RETRY') -> List[Any]:
            """Fetch all urls, then return soups for all results."""
            from bs4 import BeautifulSoup
            
            max_items = math.ceil(self.scrapping_factor*len(urls))
            if mode == 'RETRY':
                urls = random.sample(urls, k = max_items)
            
            results = asyncio.run(self.fetch_all(urls))
            
            final_results = []
            for i, result in enumerate(results):
                url = urls[i]
                if parser is None:
                    if url.endswith(".xml"):
                        parser = "xml"
                    else:
                        parser = self.default_parser
                    self._check_parser(parser)
                final_results.append(BeautifulSoup(result, parser))

            return final_results
    
    def load(self, mode) -> List[Document]:
        """Load sitemap."""
        if self.is_local:
            try:
                import bs4
            except ImportError:
                raise ImportError(
                    "beautifulsoup4 package not found, please install it"
                    " with `pip install beautifulsoup4`"
                )
            fp = open(self.web_path)
            soup = bs4.BeautifulSoup(fp, "xml")
        else:
            soup = self.scrape("xml")

        els = self.parse_sitemap(soup, mode)

        if self.blocksize is not None:
            elblocks = list(_batch_block(els, self.blocksize))
            blockcount = len(elblocks)
            if blockcount - 1 < self.blocknum:
                raise ValueError(
                    "Selected sitemap does not contain enough blocks for given blocknum"
                )
            else:
                els = elblocks[self.blocknum]

        results = self.scrape_all([el["loc"].strip() for el in els if "loc" in el])

        return [
            Document(
                page_content=self.parsing_function(results[i]),
                metadata=self.meta_function(els[i], results[i]),
            )
            for i in range(len(results))
        ]


from typing import Any, Callable, List, Optional
import re
from langchain.schema import Document
#from langchain.document_loaders.sitemap import SitemapLoader
from sitemap import SitemapLoader


class SitemapChunkLoader(SitemapLoader):
    def __init__(
        self,
        web_path: str,
        filter_urls: Optional[List[str]] = None,
        parsing_function: Optional[Callable] = None,
        blocksize: Optional[int] = None,
        blocknum: int = 0,
        meta_function: Optional[Callable] = None,
        is_local: bool = False,
    ):
        super().__init__(web_path, filter_urls, parsing_function, 
                         blocksize, blocknum,
                         meta_function, is_local)
        self._els = self._init_els()

    async def _fetch(
        self, url: str, retries: int = 3, cooldown: int = 2, backoff: float = 1.5
    ) -> str:
        print(f"Fetching {url}")
        return await super()._fetch(url, retries, cooldown, backoff)

    def _init_els(self):
        print("Loading sitemap")
        soup = self.scrape("xml")
        _els = self.parse_sitemap(soup)
        els = []

        print(f"Filtering urls {self.filter_urls}")
        locs = set()
        for el in _els:
            loc = el["loc"].strip()
            if loc in locs:
                print(f"{loc} => ALREADY COMPUTED")
                continue
            skip = False
            for r in self.filter_urls:
                if re.match(r, loc):
                    print(f"{loc} => SKIP BASED ON FILTER")
                    skip = True
                    break
            if skip is False:
                print(f"{loc} => OK")
                locs.add(loc)
                els.append(el)
        return els

    def _pop(self, size=500):
        els = []
        i = 0
        while i < size:
            try:
                el = self._els.pop()
                els.append(el)
                i += 1
            except IndexError:
                break
        return els

    def load_chunks(self, chunk_size: int = 200) -> List[Document]:
        """Load sitemap in chunks."""
        if len(self._els) <= 0:
            print("No more documents to load")
            return []

        print(f"Loading {chunk_size} documents from sitemap")
        els = self._pop(chunk_size)
        print(f"Found {len(els)} documents to load from sitemap")
        results = self.scrape_all([el["loc"].strip() for el in els if "loc" in el])
        docs = [
            Document(
                page_content=self.parsing_function(results[i]),
                metadata=self.meta_function(els[i], results[i]),
            )
            for i in range(len(results))
        ]
        print(f"Loaded {len(docs)} documents from sitemap")
        print(f"{len(self._els)} documents left in sitemap")
        return docs


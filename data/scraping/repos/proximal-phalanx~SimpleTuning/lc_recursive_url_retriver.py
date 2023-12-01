import re
from typing import Callable, Iterator, List, Optional, Set, Union

from langchain.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain.document_loaders.base import Document

from simpletuning.data_retriver import DataRetriver


class LcRecursiveUrlRetriver(DataRetriver):
    """Loads all child links from a given url."""

    url: str
    exclude_dirs: Optional[str]
    extractor: Callable[[str], str]
    max_depth: int
    timeout: int
    prevent_outside: bool

    def __init__(
        self,
        url: str,
        exclude_dirs: Optional[str] = None,
        extractor: Callable[[str], str] = lambda x: x,
        max_depth: int = 2,
        timeout: int = 10,
        prevent_outside: bool = True,
    ) -> None:
        """Initialize with URL to crawl and any subdirectories to exclude.
        Args:
            url: The URL to crawl.
            exclude_dirs: A list of subdirectories to exclude.
            use_async: Whether to use asynchronous loading,
            if use_async is true, this function will not be lazy,
            but it will still work in the expected way, just not lazy.
            extractor: A function to extract the text from the html,
            when extract function returns empty string, the document will be ignored.
            max_depth: The max depth of the recursive loading.
            timeout: The timeout for the requests, in the unit of seconds.
        """

        self.url = url
        self.exclude_dirs = exclude_dirs
        self.extractor = extractor
        self.max_depth = max_depth
        self.timeout = timeout
        self.prevent_outside = prevent_outside

    def retrive(self) -> List[Document]:
        return RecursiveUrlLoader(
            self.url,
            self.max_depth,
            True,
            self.extractor,
            self.exclude_dirs,
            self.timeout,
            self.prevent_outside,
        ).load()

import os
import tempfile
from abc import ABC
from typing import Iterator, List, Optional, Union, Dict
from urllib.parse import urlparse

import requests
from langchain.document_loaders.blob_loaders import Blob
from langchain.document_loaders.parsers import PyPDFParser
from langchain.schema import Document
from langchain.document_loaders.base import BaseLoader


class MyBasePDFLoader(BaseLoader, ABC):
    """Base Loader class for `PDF` files.

    If the file is a web path, it will download it to a temporary file, use it, then
        clean up the temporary file after completion.
    """

    def __init__(self, file_path: str, *, headers: Optional[Dict] = None, verbose: bool = False):
        """Initialize with a file path.

        Args:
            file_path: Either a local, S3 or web path to a PDF file.
            headers: Headers to use for GET request to download a file from a web path.
        """
        self.file_path = file_path
        self.web_path = None
        self.headers = headers
        self.verbose = verbose

        if "~" in self.file_path:
            self.file_path = os.path.expanduser(self.file_path)

        # If the file is a web path or S3, download it to a temporary file, and use that
        if not os.path.isfile(self.file_path) and self._is_valid_url(self.file_path):
            self.temp_dir = tempfile.TemporaryDirectory()
            _, suffix = os.path.splitext(self.file_path.split('?')[0])
            print(f"suffix: {suffix}") if verbose else None
            temp_pdf = os.path.join(self.temp_dir.name, f"tmp{suffix}")
            print(f"temp_pdf: {temp_pdf}") if verbose else None
            self.web_path = self.file_path
            if not self._is_s3_url(self.file_path):
                r = requests.get(self.file_path, headers=self.headers)
                if r.status_code != 200:
                    raise ValueError(
                        "Check the url of your file; returned status code %s"
                        % r.status_code
                    )

                with open(temp_pdf, mode="wb") as f:
                    f.write(r.content)
                self.file_path = str(temp_pdf)
        elif not os.path.isfile(self.file_path):
            raise ValueError("File path %s is not a valid file or url" % self.file_path)

    def __del__(self) -> None:
        if hasattr(self, "temp_dir"):
            self.temp_dir.cleanup()

    @staticmethod
    def _is_valid_url(url: str) -> bool:
        """Check if the url is valid."""
        parsed = urlparse(url)
        return bool(parsed.netloc) and bool(parsed.scheme)

    @staticmethod
    def _is_s3_url(url: str) -> bool:
        """check if the url is S3"""
        try:
            result = urlparse(url)
            if result.scheme == "s3" and result.netloc:
                return True
            return False
        except ValueError:
            return False

    @property
    def source(self) -> str:
        return self.web_path if self.web_path is not None else self.file_path


class MyPyPDFLoader(MyBasePDFLoader):
    """Load `PDF using `pypdf` and chunks at character level.

    Loader also stores page numbers in metadata.
    """

    def __init__(
            self,
            file_path: str,
            password: Optional[Union[str, bytes]] = None,
            headers: Optional[Dict] = None,
            verbose: bool = False,
    ) -> None:
        """Initialize with a file path."""
        try:
            import pypdf  # noqa:F401
        except ImportError:
            raise ImportError(
                "pypdf package not found, please install it with " "`pip install pypdf`"
            )
        self.parser = PyPDFParser(password=password)
        super().__init__(file_path, headers=headers, verbose=verbose)

    def load(self) -> List[Document]:
        """Load given path as pages."""
        return list(self.lazy_load())

    def lazy_load(
            self,
    ) -> Iterator[Document]:
        """Lazy load given path as pages."""
        blob = Blob.from_path(self.file_path)
        yield from self.parser.parse(blob)

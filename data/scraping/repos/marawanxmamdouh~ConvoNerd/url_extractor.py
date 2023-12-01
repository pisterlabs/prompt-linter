# Importing the necessary libraries
from http.client import InvalidURL
from typing import Iterable

import validators
from langchain.document_loaders.url import UnstructuredURLLoader
from langchain.schema import Document
from loguru import logger as log


class URLTextExtractor:
    """Class for extracting text from one or multiple URLs."""

    def __init__(self, urls: list[str]):
        """
        Initialize the extractor, specifying the urls to extract from.

        Parameters
        ----------
        urls: list[str]
            A list of URLs from which the text should be extracted.
        """
        self.urls: list[str] = urls
        self.extracted_text: list[Document] = []

    def validate_urls(self) -> Iterable[str]:
        """
        Validates the URLs provided, logs a warning and removes invalid URLs.

        Raises
        ------
        InvalidURL
            If all provided URLs are invalid.

        Returns
        -------
        valid_urls : Iterable[str]
            A list of all valid URLs.
        """
        invalid_urls: list[str] = []
        for url in self.urls:
            if not validators.url(url):
                invalid_urls.append(url)
                log.warning(f'Invalid URL: {url}')

        if len(invalid_urls) == len(self.urls):
            raise InvalidURL("All URLs are invalid")

        # Return the list of valid URLs
        return list(set(self.urls) - set(invalid_urls))

    def extract_text_from_url(self, url: str) -> list[Document]:
        """
        Uses an UnstructuredURLLoader to extract text from a single URL.

        Parameters
        ----------
        url: str
            The URL from which to extract the text.

        Returns
        -------
        extracted_text : list[Document]
            The text extracted from the URL.
        """
        unstructured_loader: UnstructuredURLLoader = UnstructuredURLLoader([url])
        return unstructured_loader.load()

    def handle_no_text_extracted(self) -> None:
        """
        Logs a warning and raises an exception if no text was extracted from the URLs.

        Raises
        ------
        InvalidURL
            If no valid content can be found in the provided URLs.
        """
        if not self.extracted_text:
            log.warning("No content was extracted from the URLs provided")
            raise InvalidURL("No valid content found in the provided URLs")

    def handle_extracted_text(self, url: str, extracted_text: list[Document]) -> None:
        """
        Appends extracted a text to the extractor's content and logged a warning if no text was extracted.

        Parameters
        ----------
        url: str
            The URL from which the text was extracted.
        extracted_text: list[Document]
            The extracted text.

        Returns
        -------
        None
            If no text was extracted.

        Warnings
        --------
        No content was extracted from the URL {url}
            If no text was extracted from the URL.

        """
        if extracted_text:
            self.extracted_text.extend(extracted_text)
        else:
            log.warning(f'No content was extracted from the URL {url}')

    def extract_text_from_urls(self) -> list[Document]:
        """
        Extracts text from all valid URLs provided and handles exceptions.

        Returns
        -------
        extracted_text: list[Document]
            A list of all texts extracted from the URLs.

        Raises
        ------
        InvalidURL
            If no valid content can be found in the provided URLs.
        """
        valid_urls: Iterable[str] = self.validate_urls()

        for url in valid_urls:
            extracted_text: list[Document] = self.extract_text_from_url(url)
            self.handle_extracted_text(url, extracted_text)

        self.handle_no_text_extracted()

        return self.extracted_text

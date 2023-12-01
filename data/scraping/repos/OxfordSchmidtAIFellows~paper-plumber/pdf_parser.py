"""Abstract base class to parse PDFs."""

import os

from langchain.document_loaders import PyPDFium2Loader

from paperplumber.logger import get_logger

logger = get_logger(__name__)


class PDFParser:
    """
    PDFParser is a class for parsing PDF documents.

    This class handles PDF parsing using the backend specified (default is "pdfium2").

    Attributes:
    _backend (str): The backend to use for PDF parsing. Default is "pdfium2".
    _pdf_path (str): The path to the PDF file to parse.
    _loader: The PDF loader instance for the specified backend.
    _pages: The list of pages obtained from the parsed PDF file.

    """

    _backend: str = "pdfium2"

    def __init__(self, pdf_path: str) -> None:
        """
        Initialize a new instance of the PDFParser class.

        Parameters:
        pdf_path (str): The path to the PDF file to parse.

        Raises:
        FileNotFoundError: If the specified file does not exist.
        """

        self._pdf_path = pdf_path

        # Check if the pdf exists
        if not os.path.exists(self._pdf_path):
            logger.error("File %s does not exist", str(self._pdf_path))
            raise FileNotFoundError(f"File {self._pdf_path} does not exist")

        # Load and split the pdf into pages
        self._loader = self._get_loader(self._backend)(pdf_path)
        self._pages = self._loader.load_and_split()

    def _get_loader(self, backend: str):
        """
        Retrieves the PDF loader class for the specified backend.

        Parameters:
        backend (str): The backend to use for PDF parsing.

        Returns:
        A class of the specified backend.

        Raises:
        ValueError: If an invalid backend is specified.
        """

        if backend == "pdfium2":
            return PyPDFium2Loader
        raise ValueError("Invalid backend")

    @property
    def pages(self):
        """
        Returns the split pages of the pdf file.

        Returns:
        A list containing the split pages of the PDF file.
        """
        return self._pages

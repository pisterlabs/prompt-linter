# Importing the necessary libraries
import os
import shutil
from typing import Iterable

from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.schema import Document

from utils.helpers import get_config

# Get the configuration
cfg = get_config('paths.yaml')


class PDFTextExtractor:
    """Class for extracting text from PDF files."""

    def __init__(self, pdf_folder_path: str = cfg.pdf_dir_path, text_folder_path: str = cfg.txt_dir_path):
        """
        Initializes the extractor, specifying the folders where the PDF and text files are located.

        Parameters
        ----------
        pdf_folder_path: str, optional
            The folder path where the PDF files are located. Default is "./uploaded_files/pdf".
        text_folder_path: str, optional
            The folder path where the extracted text files will be saved. Default is "./uploaded_files/txt".
        """
        self.pdf_folder_path: str = pdf_folder_path
        self.text_folder_path: str = text_folder_path

    def extract_text(self) -> Iterable:
        """
        Extracts the text from the PDF files located in the PDF folder.

        Returns
        -------
        Iterable
            A iterable containing the extracted text from each PDF file.
        """
        self.clear_text_folder()
        extracted_text = self.extract_text_from_pdf()
        return extracted_text

    def extract_text_from_pdf(self) -> list[Document]:
        """
        Uses a PyPDFDirectoryLoader to load and extract text from PDF files in the specified directory.

        Returns
        -------
        list
            A list of documents with the extracted text.
        """
        loader: PyPDFDirectoryLoader = PyPDFDirectoryLoader(self.pdf_folder_path)
        docs: list[Document] = loader.load()
        return docs

    def clear_text_folder(self) -> None:
        """Deletes the text folder if it exists, then creates a new one."""
        if os.path.exists(self.text_folder_path):
            shutil.rmtree(self.text_folder_path)
        os.makedirs(self.text_folder_path)

# Importing the necessary libraries
from langchain.document_loaders import TextLoader
from langchain.schema import Document
from loguru import logger as log

from utils.helpers import get_config

# Get the configuration
cfg = get_config('paths.yaml')


class TextFileExtractor:
    """Class for extracting text from a text or markdown file."""

    def __init__(self, txt_folder_path: str = cfg.txt_dir_path, md_folder_path: str = cfg.md_dir_path):
        """
        Initialize the extractor, specifying the folders where the text and markdown files are located.

        Parameters
        ----------
        txt_folder_path: str, optional.
            The folder path where the text files are located. Default is "./uploaded_files/txt".
        md_folder_path: str, optional.
            The folder path where the markdown files are located. Default is "./uploaded_files/md".
        """
        self.txt_folder_path: str = txt_folder_path
        self.md_folder_path: str = md_folder_path

    def extract_text(self, file_name: str) -> list[Document]:
        """
        Extracts the text from a given file.

        Parameters
        ----------
        file_name : str
            The name of the file from which to extract the text.

        Returns
        -------
        str
            The extracted text.

        Raises
        ------
        FileNotFoundError
            If no files were found.
        """
        if file_name:
            file_path: str = self.get_file_path(file_name)
            loader: TextLoader = TextLoader(file_path)
            extracted_text: list[Document] = loader.load()
            return extracted_text
        else:
            log.error("No files found.")
            raise FileNotFoundError("No files found.")

    def get_file_path(self, file_name: str) -> str:
        """
        Get the file path based on the file type (text or markdown).

        Parameters
        ----------
        file_name : str
            The name of the file.

        Returns
        -------
        str
            The file path.

        Raises
        ------
        TypeError
            If the file type is not supported.
        """
        if file_name.endswith(".txt"):
            return self.txt_folder_path + "/" + file_name
        elif file_name.endswith(".md"):
            return self.md_folder_path + "/" + file_name
        else:
            log.error("File type not supported.")
            raise TypeError("File type not supported.")

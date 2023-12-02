import logging
from langchain.document_loaders import UnstructuredPowerPointLoader
import tempfile
from langchain.schema.document import Document
from typing import Tuple, List


def process_pptx(file_path: str) -> List[Document]:
    """
    Process a .pptx file, extract text, and return the extracted text as a list of Document objects.

    Parameters:
    - file_path (str): The local path to the .pptx file to be processed.

    Returns:
    - List[Document]: A list of Document objects, where each object represents a slide of the .pptx with its content.

    Raises:
    - Exception: Propagates any exceptions that occur during the processing of the .pptx.
    """
    try:
        logging.info(f"Processing PPTX file: {file_path}")

        # Process the .pptx
        loader = UnstructuredPowerPointLoader(file_path)
        slides = loader.load()

        logging.info(f"PPTX processed successfully: {file_path}")

        return slides

    except Exception as e:
        logging.error(f"Error processing the PPTX: {e}")
        raise Exception(f"Error processing the PPTX: {e}")

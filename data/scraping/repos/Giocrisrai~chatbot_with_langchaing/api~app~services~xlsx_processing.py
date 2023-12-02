import logging
from typing import List
from langchain.document_loaders import UnstructuredExcelLoader
from langchain.schema.document import Document


def process_xlsx(file_path: str) -> List[Document]:
    """
    Process a local XLSX file, extract text, and return the extracted text as a list of Document objects.

    Parameters:
    - file_path (str): The local path to the XLSX file to be processed.

    Returns:
    - List[Document]: A list of Document objects, where each object represents the content of the XLSX file.

    Raises:
    - Exception: Propagates any exceptions that occur during the processing of the XLSX.
    """
    try:
        logging.info(f"Processing XLSX file: {file_path}")

        # Process the XLSX
        loader = UnstructuredExcelLoader(file_path, mode="elements")
        documents = loader.load()

        logging.info(f"XLSX processed successfully: {file_path}")

        # Return the extracted documents as a list
        return documents

    except Exception as e:
        logging.error(f"Error processing the XLSX: {e}")
        raise Exception(f"Error processing the XLSX: {e}")

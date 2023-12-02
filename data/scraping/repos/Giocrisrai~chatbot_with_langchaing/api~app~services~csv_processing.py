import logging
from typing import List
from langchain.document_loaders.csv_loader import UnstructuredCSVLoader
from langchain.schema.document import Document


def process_csv(file_path: str) -> List[Document]:
    """
    Process a local CSV file, extract table data, and return it as a list of Document objects.

    Parameters:
    - file_path (str): The local path to the CSV file to be processed.

    Returns:
    - List[Document]: A list of Document objects, where each object represents a table extracted from the CSV.

    Raises:
    - Exception: Propagates any exceptions that occur during the processing of the CSV.
    """
    try:
        logging.info(f"Processing CSV file: {file_path}")

        # Process the CSV using UnstructuredCSVLoader in "elements" mode
        loader = UnstructuredCSVLoader(file_path=file_path, mode="elements")
        documents = loader.load()

        logging.info(f"CSV processed successfully: {file_path}")

        # Return the list of Document objects representing tables
        return documents

    except Exception as e:
        logging.error(f"Error processing the CSV: {e}")
        raise Exception(f"Error processing the CSV: {e}")

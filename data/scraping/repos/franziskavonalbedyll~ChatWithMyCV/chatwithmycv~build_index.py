"""
Create and store word embeddings.

Usage:
- Specify the path of the document and the directory for the database.
- Use the `create_embeddings` function to generate and store the embeddings.

Example:
    >>> path = "<path_to_document>/document.txt"
    >>> db_dir = "embeddings_database_directory"
    >>> create_embeddings(path, db_dir)
"""

from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.document import Document
from langchain.vectorstores import Chroma


def create_embeddings(doc: Document, db_dir: str) -> None:
    """Create embeddings from document.

    :param path: path to the document
    :param db_dir: directory to store the database
    :return: None
    """
    # With Chroma we can store and retrieve word embeddings in a database
    embedding_database = Chroma(
        "cv_embeddings",
        embedding_function=OpenAIEmbeddings(),
        persist_directory=db_dir,
    )

    embedding_database.add_texts([doc.page_content], metadatas=[doc.metadata])
    embedding_database.persist()

    return embedding_database

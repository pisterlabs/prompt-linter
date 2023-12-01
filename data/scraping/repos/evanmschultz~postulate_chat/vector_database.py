from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.base import Embeddings

from langchain.vectorstores import Chroma


class VectorDatabase:
    """
    Manages the vector database for storing and querying document embeddings.
    """

    def __init__(
        self,
        persist_directory: str = "docs/chroma_db",
        embedding_function: Embeddings = OpenAIEmbeddings(client=None, chunk_size=2400),
    ) -> None:
        """
        Initializes the VectorDatabase with a persistent directory and sets up the embedding function
        using defaults.

        __init__ Args:
            persist_directory (str): The directory where the vector database should be persisted.
            embedding_function (Embeddings): The embedding function to use for the vector database.
        """
        self.persist_directory = persist_directory
        self.embedding_function = embedding_function
        # Initialize or load existing Chroma database
        self.database = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_function,
        )

    def add_documents(self, documents) -> None:
        """
        Adds documents to the vector database in determined batch sizes to avoid ChromaDB rate limits.

        Args:
            documents: The documents to add.

        Returns:
            None
        """
        batch_size = 2000
        for i in range(0, len(documents), batch_size):
            self.database.add_documents(documents=documents[i : i + batch_size])

    def query(self, query_text) -> list:
        """
        Queries the vector database with the given text and returns the top relevant documents.

        Args:
            query_text (str): The query text.

        Returns:
            list: The list of top relevant documents.
        """
        return self.database.max_marginal_relevance_search(query=query_text, k=5)

    # def update_document(self, doc_id, new_doc):
    #     self.db.update_document(doc_id, new_doc)

    # def delete_document(self, doc_id):
    #     self.db._collection.delete(ids=[doc_id])

    # def get_document(self, doc_id):
    #     return self.db._collection.get(ids=[doc_id])

    def count(self) -> int:
        """
        Counts the number of documents in the vector database.

        Returns:
            int: The number of documents.
        """
        return self.database._collection.count()

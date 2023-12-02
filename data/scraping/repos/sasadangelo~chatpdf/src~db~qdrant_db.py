from src.db.db import Database
from langchain.vectorstores import Qdrant

class QdrantDatabase(Database):
    def __init__(self):
        self.qdrant = None

    def get_context(self, question):
        context = None
        if self.qdrant:
            context = [c.page_content for c in self.qdrant.similarity_search(question, k=10)]
        return context

    def store(self, chunks, embeddings):
        if chunks:
            self.qdrant = Qdrant.from_texts(
                chunks,
                embeddings,
                path=":memory:",
                collection_name="my_collection",
                force_recreate=True
            )

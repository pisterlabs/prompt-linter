# based on: https://python.langchain.com/en/latest/modules/indexes/vectorstores/examples/pgvector.html
from typing import List, Tuple
from langchain.embeddings.openai import OpenAIEmbeddings
import langchain.vectorstores.pgvector


class RepoSearcher:
    store: langchain.vectorstores.pgvector.PGVector

    def __init__(self, collection_name: str, connection_string: str):
        self.store = langchain.vectorstores.pgvector.PGVector(
            embedding_function=OpenAIEmbeddings(),  # type: ignore
            collection_name=collection_name,
            connection_string=connection_string,
            distance_strategy=langchain.vectorstores.pgvector.DistanceStrategy.COSINE,
        )

    def find_repos(self, query: str, limit=4) -> List[Tuple[str, str]]:
        results = self.store.similarity_search_with_score(query, limit)
        # sort by relevance, returning most relevant repository first
        results.sort(key=lambda a: a[1], reverse=True)
        return [
            (r[0].metadata["namespace"], r[0].metadata["repository"]) for r in results
        ]

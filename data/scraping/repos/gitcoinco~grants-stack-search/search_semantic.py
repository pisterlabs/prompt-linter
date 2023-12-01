import time
import logging
from langchain.vectorstores import Chroma
from typing import List
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from src.util import InputDocument
from src.search import SearchEngine, SearchEngineResult, SearchType

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


class SemanticSearchEngine(SearchEngine):
    db: Chroma

    def load(self, persist_directory: str) -> None:
        logging.debug(
            "attempting to load persisted chromadb from %s", persist_directory
        )

        persisted_db = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_function,
            # Prevents negative scores and consequent UserWarning. See https://github.com/langchain-ai/langchain/issues/10864
            collection_metadata={"hnsw:space": "cosine"},
        )

        sample = persisted_db.get(limit=1)
        if len(sample["ids"]) == 0:
            raise Exception(
                f"Could not load persisted chromadb from {persist_directory}"
            )
        else:
            logging.debug("chromadb loaded from %s", persist_directory)

        self.db = persisted_db

    def index(
        self,
        project_docs: List[InputDocument],
        persist_directory: str | None = None,
    ) -> None:
        start_time = time.perf_counter()

        raw_documents = [project_document.document for project_document in project_docs]

        self.db = Chroma.from_documents(
            documents=raw_documents,
            embedding=embedding_function,
            ids=[doc.metadata["application_ref"] for doc in raw_documents],
            persist_directory=persist_directory,
            # Prevents negative scores and consequent UserWarning. See https://github.com/langchain-ai/langchain/issues/10864
            collection_metadata={"hnsw:space": "cosine"},
        )

        logging.debug(
            "indexed %d projects in %.2f seconds with chroma",
            len(project_docs),
            time.perf_counter() - start_time,
        )

        self.db.persist()

    def search(
        self, query_string: str, min_score: float = 0.35
    ) -> List[SearchEngineResult]:
        return [
            SearchEngineResult(
                ref=raw_doc.metadata["application_ref"],
                score=score,
                type=SearchType.semantic,
            )
            for raw_doc, score in self.db.similarity_search_with_relevance_scores(
                query_string,
                k=100,
            )
            if score > min_score
        ]

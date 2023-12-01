from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.embeddings.base import Embeddings
from langchain.embeddings import OpenAIEmbeddings
import uuid
from typing import Any, List, Optional, Tuple
import numpy as np
from bot.models import SourceDocument

def dependable_faiss_import() -> Any:
    """Import faiss if available, otherwise raise error."""
    try:
        import faiss
    except ImportError:
        raise ValueError(
            "Could not import faiss python package. "
            "Please install it with `pip install faiss` "
            "or `pip install faiss-cpu` (depending on Python version)."
        )
    return faiss



class PostgresFAISS(FAISS):
    def similarity_search_with_score_by_vector(
        self, embedding: List[float], k: int = 4
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.
        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
        Returns:
            List of Documents most similar to the query and score for each
        """
        scores, indices = self.index.search(np.array([embedding], dtype=np.float32), k)
        docs = []
        for j, i in enumerate(indices[0]):
            if i == -1:
                # This happens when not enough docs are returned.
                continue
            _id = self.index_to_docstore_id[i]
            try:
                dbDoc = SourceDocument.objects.get(id=_id)
            except SourceDocument.DoesNotExist:
                raise ValueError(f"Could not find document for id {_id}")
            doc = Document(page_content=dbDoc.content, metadata={'source': dbDoc.id})
            docs.append((doc, scores[0][j]))
        return docs

    @classmethod
    def from_query(
        cls,
        query
    ):
        faiss = dependable_faiss_import()
        embeddings = []
        index_to_id = {}
        i = 0
        for doc in query:
            embeddings.append(doc.embedding)
            index_to_id[i] = doc.id
            i += 1
        
        embeddings = np.array(embeddings, dtype=np.float32)
        index = faiss.IndexFlatL2(len(embeddings[0]))
        index.add(np.array(embeddings, dtype=np.float32))
        return cls(OpenAIEmbeddings().embed_query, index, None, index_to_id)

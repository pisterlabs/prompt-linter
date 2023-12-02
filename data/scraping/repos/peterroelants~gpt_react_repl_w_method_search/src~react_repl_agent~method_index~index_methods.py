import numpy as np
import openai
from openai.embeddings_utils import cosine_similarity

from .doc_utils import MethodDoc


def get_embedding(text: str):
    """Get embedding of given text using OpenAI's text embedding API."""
    response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
    embeddings = response["data"][0]["embedding"]
    return embeddings


class MethodsVectorIndex:
    """Index of method docs using vector embeddings."""

    def __init__(
        self,
        method_docs: list[MethodDoc],
    ):
        self.method_docs = method_docs
        # Only index method name and descriptions for now
        doc_strs = [
            f"`{method_doc.name}`: {method_doc.description}".strip()
            for method_doc in self.method_docs
        ]
        self.doc_embeddings = [get_embedding(s) for s in doc_strs]

    def search(
        self,
        query: str,
        topk: int,
    ) -> tuple[list[MethodDoc], list[int], list[float]]:
        """Search for methods that match the given query."""
        query_embedding = get_embedding(query)
        cos_sims = [
            cosine_similarity(query_embedding, doc_embedding)
            for doc_embedding in self.doc_embeddings
        ]
        sorted_idxs = np.argsort(cos_sims)[::-1]
        assert len(sorted_idxs) == len(cos_sims)
        assert cos_sims[sorted_idxs[0]] >= cos_sims[sorted_idxs[-1]]
        top_k_idxs = sorted_idxs[:topk].tolist()
        assert len(top_k_idxs) == topk
        sims = [cos_sims[i] for i in top_k_idxs]
        selected_docs = [self.method_docs[i] for i in top_k_idxs]
        return (selected_docs, top_k_idxs, sims)

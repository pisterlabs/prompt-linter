from typing import Any, Dict, List, Optional, Union

import numpy as np
import firebase_admin
from firebase_admin import firestore
from google.cloud import aiplatform
from langchain.schema import BaseRetriever, Document
from langchain.callbacks.manager import CallbackManagerForRetrieverRun, AsyncCallbackManagerForRetrieverRun
from langchain.embeddings.base import Embeddings
from langchain.embeddings import TensorflowHubEmbeddings


# Application Default credentials are automatically created.
app = firebase_admin.initialize_app()
db = firestore.client()


class FirestoreRetriever(BaseRetriever):

    index_endpoint_name: str
    deployed_index_id: str
    embeddings: Embeddings
    collection: str
    top_k: int = 5

    def _similarity_search(self, query_emb: np.ndarray):
        """
        Perform a similarity search.

        Args:
            query_emb: Query represented as an embedding

        Returns:
            A list of documents most similar to the query
        """
        my_index_endpoint = aiplatform.MatchingEngineIndexEndpoint(
            index_endpoint_name=self.index_endpoint_name
        )

        similar_docs = my_index_endpoint.find_neighbors(
            deployed_index_id=self.deployed_index_id,
            queries=query_emb,
            num_neighbors=self.top_k
        )

        return similar_docs

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        query_embedding = self.embeddings.embed_documents([query])
        similar_docs = self._similarity_search(query_embedding)

        relevant_docs = []
        for doc in similar_docs[0]:
            doc_id = doc.id
            doc_ref = db.collection(self.collection).document(doc_id)

            doc = doc_ref.get()
            relevant_docs.append(self._firestore_doc_to_langchain_doc(doc))
        return relevant_docs


    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        raise NotImplementedError()

    def _firestore_doc_to_langchain_doc(self, fs_doc) -> Document:
        lc_doc = Document(
            page_content=fs_doc.get("context")
        )
        return lc_doc

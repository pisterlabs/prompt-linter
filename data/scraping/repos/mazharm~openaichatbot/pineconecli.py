"""
This module is a helper library to connect to the Pinecone API
"""
import os
from pinecone import init, Index
from .openaicli import OpenAICli


class PineconeCli:
    """
    This class is used to interact with the Pinecone API
    """
    def __init__(self):
        # Set up Pinecone API credentials
        init(api_key=os.getenv("PINECONE_API_KEY"), environment="us-east1-gcp")
        self.oac = OpenAICli()

    def upsert_vectors(self, vectors, index_name="openai-embeddings"):
        """
        Upload the embeddings to Pinecone
        """
        pinecone_index = Index(index_name=index_name)

        # Store the embeddings and associated facts in Pinecone
        pinecone_index.upsert(vectors=vectors)

    def find_match(self, text, n=10, index_name="openai-embeddings"):
        """
        Find the closest match in Pinecone
        """
        pinecone_index = Index(index_name=index_name)
        query_vector = self.oac.get_embedding(text, "text-embedding-ada-002")
        results = pinecone_index.query(vector=query_vector, top_k=n, include_metadata=True)

        match = ""

        for result in results['matches']:
            if result['score'] > 0.7:
                match += result['metadata']['fact'] + "\n"

        return match

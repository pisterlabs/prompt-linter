
from langchain.embeddings.base import Embeddings
from typing import List
from sentence_transformers import SentenceTransformer
import os

# Create Open API key here : https://platform.openai.com/account/api-keys.
open_api_key = os.getenv('OPEN_API_KEY')
default_hf_embedding = 'multi-qa-mpnet-base-dot-v1'
FAISS_INDEX_PATH = 'knowledge_faiss_index'

class CustomHuggingFaceEmbeddings(Embeddings):
    def __init__(self, model_id): 
        # Should use the GPU by default
        self.model = SentenceTransformer(model_id)
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using a locally running
           Hugging Face Sentence Transformer model
        Args:
            texts: The list of texts to embed.
        Returns:
            List of embeddings, one for each text.
        """
        embeddings =self.model.encode(texts)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a query using a locally running HF 
        Sentence trnsformer. 
        Args:
            text: The text to embed.
        Returns:
            Embeddings for the text.
        """
        embedding = self.model.encode(text)
        return list(map(float, embedding))
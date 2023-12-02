from typing import Any, Dict, List, Optional

from langchain.embeddings.huggingface import HuggingFaceEmbeddings, DEFAULT_MODEL_NAME
from langchain.pydantic_v1 import Field

class CustomSTEmbeddings(HuggingFaceEmbeddings):
    '''
        Same as the basic HuggingFaceEmbeddings, 
        but with a configurable embedding dimension for better integration with Pinecone
    '''
    client: Any  #: :meta private:
    model_name: str = DEFAULT_MODEL_NAME
    """Model name to use."""
    cache_folder: Optional[str] = None
    """Path to store models. 
    Can be also set by SENTENCE_TRANSFORMERS_HOME environment variable."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Key word arguments to pass to the model."""
    encode_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Key word arguments to pass when calling the `encode` method of the model."""
    multi_process: bool = False
    """Run encode() on multiple GPUs."""
    embedding_dimension: int = 1536
    '''To match the dimension of vector store db'''

    def __init__(self, **kwargs: Any):
        """
            Initialize the sentence_transformer.

            Extra input than HuggingFaceEmbeddings:
                embedding_dimension: dimension for embedded vector, default to 1536, 
                same as the OpenAI embedding
        """
        super().__init__(**kwargs)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a HuggingFace transformer model and adjust the embedding dimension.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        embeddings = super().embed_documents(texts)
        for i in range(len(embeddings)):
            while len(embeddings[i]) < self.embedding_dimension:
                embeddings[i].append(0.0)
        return embeddings
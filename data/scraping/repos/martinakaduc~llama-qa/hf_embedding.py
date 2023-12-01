import torch

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Extra, Field
from langchain.embeddings.base import Embeddings

DEFAULT_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_INSTRUCT_MODEL = "hkunlp/instructor-large"
DEFAULT_EMBED_INSTRUCTION = "Represent the document for retrieval: "
DEFAULT_QUERY_INSTRUCTION = (
    "Represent the question for retrieving supporting documents: "
)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(token_embeddings.dtype)
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

class HuggingFaceEmbeddings(BaseModel, Embeddings):
    """HuggingFace sentence_transformers embedding models.

    To use, you should have the ``sentence_transformers`` python package installed.

    Example:
        .. code-block:: python

            from langchain.embeddings import HuggingFaceEmbeddings

            model_name = "sentence-transformers/all-mpnet-base-v2"
            model_kwargs = {'device': 'cpu'}
            encode_kwargs = {'normalize_embeddings': False}
            hf = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
    """

    model: Any
    """The HuggingFace model."""
    tokenizer: Any
    """The HuggingFace tokenizer."""
    cache_folder: Optional[str] = None
    """Path to store models. 
    Can be also set by SENTENCE_TRANSFORMERS_HOME environment variable."""
    encode_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Key word arguments to pass when calling the `encode` method of the model."""

    def __init__(self, model, tokenizer, **kwargs: Any):
        """Initialize the sentence_transformer."""
        super().__init__(**kwargs)
        self.model = model
        self.tokenizer = tokenizer

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid


    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a HuggingFace transformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        texts = list(map(lambda x: x.replace("\n", " "), texts))
        embeddings = self.encode(texts, **self.encode_kwargs)
        return embeddings.tolist()


    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a HuggingFace transformer model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        text = text.replace("\n", " ")
        embedding = self.encode([text], **self.encode_kwargs)[0]
        return embedding.tolist()
    
    def encode(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        """Encode a list of texts.

        Args:
            texts: The list of texts to embed.
            kwargs: Additional keyword arguments to pass to the model.

        Returns:
            List of embeddings, one for each text.
        """
        
        #Tokenize sentences
        encoded_input = self.tokenizer(texts, padding=True, return_tensors='pt')
        
        #Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        #Perform pooling. In this case, mean pooling
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = sentence_embeddings.detach().cpu().numpy()
        
        return sentence_embeddings
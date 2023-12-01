from langchain.embeddings import HuggingFaceInstructEmbeddings
from src.utils.singleton import Singleton

class EmbeddingModel(metaclass=Singleton):
    """A singleton class that loads the embedding model for embedding text."""

    def __init__(self) -> None:
        self.model_name = 'intfloat/e5-base-v2'
        self.model_kwargs = {'device': 'cpu'}
        self.model = HuggingFaceInstructEmbeddings(model_name=self.model_name, model_kwargs=self.model_kwargs)

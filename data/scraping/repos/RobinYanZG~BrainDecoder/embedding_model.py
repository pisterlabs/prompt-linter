from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from app.core.torch_management import torchMngr
from config.config import settings

class EmbeddingModel(HuggingFaceEmbeddings):
    def __init__(self):
        super().__init__(
            model_name=settings.EMBEDDING_MODEL, 
            model_kwargs={'device': torchMngr.get_embedding_device()}
        )
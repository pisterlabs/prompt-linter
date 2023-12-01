# embedding_setup.py

from langchain.embeddings import HuggingFaceEmbeddings

from .config import get_env_variable


def setup_embeddings_model():
    """to handle the setup of the HuggingFaceEmbeddings model.
    This will help keep the embeddings-related code organized.
    """
    model_name = get_env_variable("EMBEDDINGS_MODEL_NAME")
    return HuggingFaceEmbeddings(model_name=model_name)

from langchain.embeddings.huggingface import HuggingFaceEmbeddings


def hf_embeddings(model_name_or_path: str) -> HuggingFaceEmbeddings:
    """
    Get huggingface embeddings.
    """
    return HuggingFaceEmbeddings(model_name=model_name_or_path)

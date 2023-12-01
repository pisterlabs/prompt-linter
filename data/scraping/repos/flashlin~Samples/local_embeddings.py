from chromadb import Documents, EmbeddingFunction, Embeddings
from langchain.embeddings import HuggingFaceBgeEmbeddings


class LocalEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name: str):
        embedding = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': True}
        )

    def __call__(self, texts: Documents) -> Embeddings:
        # embed the documents somehow
        return embeddings
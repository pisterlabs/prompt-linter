import torch
from langchain.embeddings import HuggingFaceEmbeddings


def get_hf_embedder():
    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    )

    return embedder

from langchain.embeddings import HuggingFaceBgeEmbeddings
import torch


def get_embedding(model_name="BAAI/bge-large-zh"):
    # 如果GPU存在，使用GPU，否则使用CPU
    model_kwargs = {'device': 'cuda'} if torch.cuda.is_available() else {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}  # set True to compute cosine similarity
    return HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

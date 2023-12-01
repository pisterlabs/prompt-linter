import numpy as np
import openai
from typing import List


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_embedding(text: str, engine="text-embedding-ada-002", **kwargs) -> List[float]:

    # replace newlines, which can negatively affect performance.
    text = text.replace("\n", " ")

    return openai.Embedding.create(input=[text], engine=engine, **kwargs)["data"][0]["embedding"]

def normalize_embeddings(embeddings):
    # Compute the Euclidean norm for each embedding.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Avoid division by zero.
    norms[norms == 0] = 1
    
    # Normalize each embedding to have a length of 1.
    normalized_embeddings = embeddings / norms
    
    return normalized_embeddings

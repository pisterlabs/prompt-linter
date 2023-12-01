import faiss
import openai
import numpy as np
from typing import List

# API 호출 모델
EMBEDDING_MODEL = 'text-embedding-ada-002'


def get_embedding(msg: str) -> List:
    embedding = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=msg,
    )
    return embedding["data"][0]["embedding"]


documents = ["안녕하세요.", "안녕히가세요.", "고마워요.", "반가워요."]
embeddings = [get_embedding(doc) for doc in documents]
embedding_matrix = np.array(embeddings)
print(embedding_matrix)
index = faiss.IndexFlatL2(embedding_matrix.shape[1])
index.add(embedding_matrix)


def find_similar_document(user_embedding: List) -> str:
    _, top_indices = index.search(np.array([user_embedding]), 1)
    top_index = top_indices[0][0]
    return documents[top_index]


res = find_similar_document(get_embedding("난 갈게. 고마워~"))
print(res)
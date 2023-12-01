"Entry point for the backend API"
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cohere
import numpy as np
import os

co = cohere.Client(os.getenv("COHERE_API_KEY"))


def calculate_similarity(val_a, val_b):
    """
    Cosine Similarity, less sensitive to a difference in lengths
    """
    return np.dot(val_a, val_b) / (np.linalg.norm(val_a) * np.linalg.norm(val_b))


class Item(BaseModel):
    """
    Data model for incoming request
    """

    posting: str
    resume: str


class Similarity(BaseModel):
    """
    return similarity
    """

    similarity: float


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/v1/similarity/")
async def root(data: Item) -> Similarity:
    """
    since we just care the similarity between job postings and resumes,
    using embeddings is enough. TODO: try BERT
    """
    (post1, post2) = co.embed(
        texts=[data.posting, data.resume],
        model="embed-english-v2.0",
    ).embeddings

    similarity = calculate_similarity(post1, post2)

    return Similarity(similarity=similarity)

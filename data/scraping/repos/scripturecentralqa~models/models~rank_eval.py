"""Functions to evaluate ranking models."""
import os
from typing import Callable
from typing import Union
from typing import cast

import cohere  # type: ignore
import numpy as np
import openai
from sentence_transformers import SentenceTransformer  # type: ignore
from sklearn.metrics import ndcg_score  # type: ignore


def get_embeddings_fn(model_name: str) -> Callable[[list[str]], list[list[float]]]:
    """Return a function that takes a list of texts and returns a list of embeddings."""
    if model_name == "text-embedding-ada-002":
        # embedding_len = 1536, metric = "cosine"
        # init openai
        openai.api_key = os.environ["OPENAI_API_KEY"]

        def get_embeddings(texts: list[str]) -> list[list[float]]:
            res = openai.Embedding.create(input=texts, engine=model_name)  # type: ignore
            return [cast(list[float], record["embedding"]) for record in res["data"]]

        return get_embeddings
    elif model_name == "embed-english-v2.0":
        cohere_client = cohere.Client(os.environ["COHERE_KEY"])

        def get_embeddings(texts: list[str]) -> list[list[float]]:
            res = cohere_client.embed(texts=texts, model=model_name, truncate="END")
            return cast(list[list[float]], res.embeddings)

        return get_embeddings
    elif model_name in [
        "all-mpnet-base-v2",
        "multi-qa-mpnet-base-dot-v1",
        "multi-qa-MiniLM-L6-cos-v1",
        "multi-qa-distilbert-cos-v1",
    ]:
        # all-mpnet-base-v2: embedding_len = 768, metric = "cosine"
        # multi-qa-mpnet-base-dot-v1: embedding_len = 768, metric = "dotproduct"
        # multi-qa-MiniLM-L6-cos-v1: embedding_len = 384, metric = "cosine"
        # multi-qa-distilbert-cos-v1: embedding_len = 768, metric = "cosine"
        model = SentenceTransformer("sentence-transformers/" + model_name)

        def get_embeddings(texts: list[str]) -> list[list[float]]:
            return cast(list[list[float]], model.encode(texts).tolist())

        return get_embeddings

    else:
        raise ValueError(f"Unknown model_name: {model_name}")


def get_ndcg(
    true_results: list[dict[str, Union[str, float]]], pred_results: list[dict[str, Union[str, float]]], k: int = 10
) -> float:
    """Return the Normalized Discounted Cumulative Gain score for two {id, score} lists."""
    all_results = list({result["id"] for result in true_results} | {result["id"] for result in pred_results})
    true_scores = [
        next((item["score"] for item in true_results if item["id"] == result), 0.0) for result in all_results
    ]
    pred_scores = [
        next((item["score"] for item in pred_results if item["id"] == result), 0.0) for result in all_results
    ]
    return cast(float, ndcg_score(np.asarray([true_scores]), np.asarray([pred_scores]), k=k))

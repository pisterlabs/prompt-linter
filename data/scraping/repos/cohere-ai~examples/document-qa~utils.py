import os
import pandas as pd
from typing import Any, List
from io import StringIO
import random
import string
import numpy as np
import cohere
from numpy.linalg import norm

co_client = cohere.Client(f'{os.environ["CO_KEY"]}')
CHUNK_SIZE = 1024
OUTPUT_BASE_DIR = "./"


def get_random_string(length: int = 10):
    letters = string.ascii_letters
    result_str = "".join(random.choice(letters) for i in range(length))
    return result_str


def process_csv_file(st_file_object: Any, run_id: str = None):
    df = pd.read_csv(StringIO(st_file_object.getvalue().decode("utf-8")))
    run_id = get_random_string() if run_id is None else run_id

    output_path = os.path.join(OUTPUT_BASE_DIR, f"{run_id}.csv")

    return df, run_id, output_path, len(df)


def process_text_input(text: str, run_id: str = None):
    text = StringIO(text).read()
    chunks = [text[i: i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
    df = pd.DataFrame.from_dict({"text": chunks})
    run_id = get_random_string() if run_id is None else run_id

    output_path = os.path.join(OUTPUT_BASE_DIR, f"{run_id}.csv")

    return df, run_id, output_path, len(df)


def embed_stuff(list_of_texts):
    response = co_client.embed(model="small", texts=list_of_texts)
    return response.embeddings


def get_embeddings_from_df(df):
    return embed_stuff(list(df.text.values))


def top_n_neighbours_indices(
    prompt_embedding: np.ndarray, storage_embeddings: np.ndarray, n: int = 5
):
    if isinstance(storage_embeddings, list):
        storage_embeddings = np.array(storage_embeddings)
    if isinstance(prompt_embedding, list):
        storage_embeddings = np.array(prompt_embedding)
    similarity_matrix = (
        prompt_embedding
        @ storage_embeddings.T
        / np.outer(norm(prompt_embedding, axis=-1), norm(storage_embeddings, axis=-1))
    )
    num_neighbours = min(similarity_matrix.shape[1], n)
    indices = np.argsort(similarity_matrix, axis=-1)[:, -num_neighbours:]

    return indices


def select_prompts(list_of_texts, sorted_indices):
    return np.take_along_axis(np.array(list_of_texts)[:, None], sorted_indices, axis=0)


def get_augmented_prompts(prompt_embedding, storage_embeddings, storage_df) -> List:
    assert prompt_embedding.shape[0] == 1
    if isinstance(prompt_embedding, list):
        prompt_embedding = np.array(prompt_embedding)
    indices = top_n_neighbours_indices(prompt_embedding, storage_embeddings, n=5)
    similar_prompts = select_prompts(storage_df.text.values, indices)

    return similar_prompts[0]

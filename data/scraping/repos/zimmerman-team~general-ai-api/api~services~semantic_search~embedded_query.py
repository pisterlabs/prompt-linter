import json
import os

import numpy as np
import openai
import pandas as pd
from openai.embeddings_utils import cosine_similarity, get_embedding

openai.api_key = os.environ.get("AIAPI_OPENAI_API_KEY")

# embedding model parameters
EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_STORAGE = "./api/services/semantic_search/data/embeddings_previous_queries.json"


def embedded_query(filepath, query):
    """
    We assume the filepath is a csv with embeddings created
    """
    storage_key = filepath + "==" + query
    # Check if the query has been made before
    with open(EMBEDDING_STORAGE, 'r') as f:
        previous_suggestions = json.load(f)
        if storage_key in previous_suggestions:
            print("Query has been made before, returning previous suggestion")
            return 200, previous_suggestions[storage_key]

    print("Query has not been made before, getting embedding")
    # If the query has not been made before, we need to get the embedding
    df = pd.read_csv(filepath)
    df["embedding"] = df.embedding.apply(eval).apply(np.array)
    # Ask the question
    res = _search_embeddings(df, query, 1).context.values[0]
    # Save the emmbedding as follows: key = filepath+query, value = res
    with open(EMBEDDING_STORAGE, 'r') as f:
        previous_suggestions = json.load(f)
        previous_suggestions[storage_key] = res
    with open(EMBEDDING_STORAGE, 'w') as f:
        json.dump(previous_suggestions, f)

    return 200, res


def _search_embeddings(df, query, n_results=3):
    embedded_query = get_embedding(
        query,
        engine=EMBEDDING_MODEL
    )
    df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, embedded_query))

    results = (
        df.sort_values("similarity", ascending=False)
        .head(n_results)
    )
    return results


def get_query_from_file(file):
    tmp_path = f"./{file.filename}"
    file.save(tmp_path)
    audio_file = open(tmp_path, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    os.remove(tmp_path)
    return transcript.text

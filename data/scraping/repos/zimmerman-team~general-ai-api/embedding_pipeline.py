import os

import openai
import pandas as pd
import tiktoken
from openai.embeddings_utils import get_embedding

openai.api_key = os.environ.get("AIAPI_OPENAI_API_KEY")

# embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191
print("get encoding when initializing the embedding pipeline instead of each time the request is made")
encoding = tiktoken.get_encoding(embedding_encoding)


def embedding_pipeline(filepath):
    print(f"Starting embedding pipeline for {filepath}")
    df = pd.read_csv(filepath)
    df = df.dropna()  # clean up the data
    # Combine all columns into one column called 'context' in a string
    df['context'] = df.apply(lambda x: ', '.join([f"{col}: {x[col]}" for col in df.columns]), axis=1)
    # Filter out entries that are too long
    df['n_tokens'] = df.context.apply(lambda x: len(encoding.encode(x)))
    df = df[df.n_tokens <= max_tokens]
    # Get the embeddings for the context
    # use the filepath and replace the .csv with _with_embeddings.csv
    embedding_filepath = filepath.replace(".csv", "_with_embeddings.csv")
    # Check if the embedding file already exists
    if os.path.isfile(embedding_filepath):
        return 200, "Embedding file already exists, skipping embedding pipeline"
    else:
        df["embedding"] = df.context.apply(lambda x: get_embedding(x, engine=embedding_model))
        df.to_csv(embedding_filepath)  # Save the embeddings

    return 200, "Embeddings generated"

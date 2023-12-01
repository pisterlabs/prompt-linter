import os
import cohere
import pandas as pd
import dotenv
import tiktoken
import numpy as np
from tenacity import retry, wait_fixed

dotenv.load_dotenv()
co = cohere.Client(os.getenv('COHERE_API_KEY'))
enc = tiktoken.get_encoding("cl100k_base")


def main():
    input_file = 'embeddings/scotus_opinions.parquet'
    output_file = 'embeddings/scotus_opinions.parquet'

    df = pd.read_parquet(input_file)

    get_embeddings(df, output_file)


def get_embeddings(df, output_file):
    """DataFrame must have columns 'text' and 'embedding'"""
    total = len(df)
    for i, row in df.iterrows():
        if row['embedding'] is not None:
            continue
        if i % 100 == 0:
            print(f'Processing row {i}, Progress: {i / total * 100:.2f}%')
            df.to_parquet(output_file)
        if row['text'] is None:
            continue
        df.at[i, 'embedding'] = get_embedding(row['text'])

    df.to_parquet(output_file)


def get_embedding(text):
    groups = split_text(text)
    embeddings = [call_cohere(group) for group in groups]

    # Flatten embeddings and groups
    embeddings = [item for sublist in embeddings for item in sublist]
    groups = [item for sublist in groups for item in sublist]

    # Calculate weighted average
    embedding = weighted_average_embedding(embeddings, groups)

    return embedding


@retry(wait=wait_fixed(10))
def call_cohere(texts):
    return co.embed(texts=texts, model="embed-english-v2.0").embeddings


def weighted_average_embedding(embeddings, groups):
    weights = np.array([len(group) for group in groups])
    embedding = np.average(np.array(embeddings), axis=0, weights=weights)
    return embedding


def split_text(text, MAX_TOKENS=512, MAX_TEXTS=96):
    # Split text into tokens
    tokens = enc.encode(text)

    # Split tokens into chunks of size MAX_TOKENS
    chunks = [enc.decode(tokens[i:i + MAX_TOKENS])
              for i in range(0, len(tokens), MAX_TOKENS)]

    # Split chunks into groups of size MAX_TEXTS
    groups = [chunks[i:i + MAX_TEXTS]
              for i in range(0, len(chunks), MAX_TEXTS)]

    return groups


if __name__ == '__main__':
    main()

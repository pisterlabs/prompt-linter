from alive_progress import alive_bar
from ast import literal_eval
from openai.embeddings_utils import get_embedding, cosine_similarity
import glob
import numpy as np
import openai
import os
import pandas as pd
import sqlite3
import sys
import tiktoken
import tomllib

with open("config.toml", "rb") as file:
    config = tomllib.load(file)

openai.api_key = config['OPENAI_KEY']
TOKEN_LIMIT = config['TOKEN_LIMIT']
EMBEDDINGS_DB = config['EMBEDDINGS_DB']
DIRS = config['DIRS']

def init_db(conn) -> None:

    # Create an empty DataFrame with specified columns and data types
    df = pd.DataFrame(columns=['file', 'part', 'modified', 'embedding'])

    # Set the data types for each column
    df['file'] = df['file'].astype('str')
    df['part'] = df['part'].astype('int64')
    df['modified'] = df['modified'].astype('float64')
    df['embedding'] = df['embedding'].astype('str')

    df.to_sql('vault', conn, index=False)

def load_df() -> pd.DataFrame:

    conn = sqlite3.connect(EMBEDDINGS_DB)

    # Check if the 'vault' table exists and init if needed
    existing_tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='vault';").fetchall()
    if not existing_tables:
        init_db(conn)

    # Read the 'vault' table into a DataFrame
    df = pd.read_sql('select * from vault', conn)

    # Convert the 'modified' column to a timestamp
    df['modified'] = df['modified'].astype('float64')

    # Remove files that have been deleted or modified
    for index, row in df.iterrows():
        if (
            not os.path.exists(row['file']) or
            os.path.getmtime(row['file']) > row['modified']
        ):
            print(f"Removing {row['file']} from index (deleted or modified)")
            df.drop(index, inplace=True)

    # Transform embedding column
    df["embedding"] = df['embedding'].apply(literal_eval).apply(np.array)

    return df


def save_df(df) -> None:

    # Convert list to string for sqlite
    df['embedding'] = df['embedding'].apply(lambda x: ','.join(map(str, x)))
    conn = sqlite3.connect(EMBEDDINGS_DB)

    # Convert the 'modified' column to a string
    df['modified'] = df['modified'].astype(str)

    df.to_sql('vault', conn, if_exists='replace', index=False)


def get_chunks(text):
    chunks = []
    current_chunk = []
    encoding = tiktoken.get_encoding('cl100k_base')
    encoded = encoding.encode(text)
    for i, token in enumerate(encoded):
        current_chunk.append(token)
        if (
            len(current_chunk) >= TOKEN_LIMIT
            or i == len(encoded) - 1
        ):
            chunks.append(current_chunk)
            current_chunk = []
            next
    return chunks


def build():

    encoding = tiktoken.get_encoding('cl100k_base')

    existing_df = load_df()
    print(f"{len(existing_df)} files already indexed")

    files = []
    for dir in DIRS:
        files.extend(glob.glob(f"{dir}/*.md"))
    print(f"{len(files)} files found")

    files = list(set(files) - set(existing_df['file']))
    print(f"{len(files)} files need embeddings")

    file_chunks = []
    file_parts = []
    mod_timestamps = []
    embeddings = []

    with alive_bar(len(files)) as bar:
        for f in files:
            print(f)
            with open(f, "r") as file:
                mtime = os.path.getmtime(f)
                chunks = get_chunks(file.read())
                for i, chunk in enumerate(chunks):
                    content = encoding.decode(chunk)
                    embeddings.append(get_embedding(content, engine='text-embedding-ada-002'))
                    file_chunks.append(f)
                    file_parts.append(i)
                    mod_timestamps.append(mtime)
            bar()

    df = pd.DataFrame({
        'file': file_chunks,
        'part': file_parts,
        'modified': mod_timestamps,
        'embedding': embeddings,
    })

    if len(existing_df) > 0:
        df = pd.concat([existing_df, df], ignore_index=True)

    save_df(df)


def search(df, query):
    query_embedding = get_embedding(query, engine='text-embedding-ada-002')
    # print(df)
    # print(query_embedding)
    df['similarity'] = df['embedding'].apply(lambda x: cosine_similarity(x, query_embedding))
    results = (
        df.sort_values("similarity", ascending=False)
        .head(3)
    )
    return results

def main():

    build()
    df = load_df()
    while True:
        cmd = input("Enter a query: ")
        if cmd.lower() == 'q':
            sys.exit()
        response = search(df, cmd)
        for index, row in response.iterrows():
            print(f"{row['file']} ({row['part']}) - {row['similarity']}")
            with open(row['file'], 'r') as file:
                content = file.read()
                print(content[:100])

if __name__ == '__main__':
    main()

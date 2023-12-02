"""
find_unique_genres.py
---------------------
Script for processing the 'combined_songs.csv' dataset to identify and list unique music genres.
Additionally, retrieves embeddings for each genre using the OpenAI API.
"""

import pandas as pd
import openai
from dotenv import load_dotenv
load_dotenv()
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

# load the genre data
df = pd.read_csv('combined_songs.csv')

# Entfernen Sie die Anführungszeichen
df['genre'] = df['genre'].str.replace('"', '')

# remove any rows with missing genre data
df['genre'].fillna('Unknown', inplace=True)

# Flattening-Operation durchführen
all_genres = [genre for sublist in df['genre'].str.split(',') for genre in sublist]

# Konvertieren Sie die Liste in ein Set, um doppelte Einträge zu entfernen
unique_genres = set(all_genres)

print(f"Number of unique genres: {len(unique_genres)}")
print(unique_genres)

# get embeddings for each genre
genre_embeddings = {}
for genre in unique_genres:
    response = openai.Embedding.create(input=genre.strip(), engine="text-embedding-ada-002")

    # Check if the response contains the embedding
    if 'data' in response and isinstance(response['data'], list) and 'embedding' in response['data'][0]:
        embedding = response['data'][0]['embedding']
        genre_embeddings[genre] = embedding
    else:
        print(f"Failed to extract embedding for genre: {genre}")

print(f"Number of genres in genre_embeddings: {len(genre_embeddings)}")

# save embeddings to CSV
embedding_df = pd.DataFrame(list(genre_embeddings.items()), columns=['Genre', 'Embedding'])
print(embedding_df)
embedding_df.to_csv('genre_embeddings.csv', index=False)

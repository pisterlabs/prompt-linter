import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
from openai.embeddings_utils import get_embedding
import numpy as np

def search_books(query):
    """Search books based on a query and print the titles of the top 3 most similar books."""
    df = load_data('book_summaries_with_embeddings.csv')
    query_embedding = calculate_embedding(query)
    df = calculate_similarities(df, query_embedding)
    print_top_books(df)

def load_data(input_path):
    """Load the embeddings from a CSV file."""
    return pd.read_csv(input_path)

def calculate_embedding(query):
    """Calculate the embedding for a query."""
    return get_embedding(query, engine="text-embedding-ada-002")

def calculate_similarities(df, query_embedding):
    """Calculate the cosine similarity between the query and the embeddings of each book."""
    embeddings = np.array(df['embedding'].tolist())
    query_embedding = np.array(query_embedding)
    df['similarity'] = 1 - cdist(df['embedding'], [query_embedding], 'cosine')
    return df.sort_values(by='similarity', ascending=False)

def print_top_books(df):
    """Print the titles of the top 3 most similar books."""
    print(df['book_id'].head(3))

if __name__ == "__main__":
    search_books('medieval sci-fi dystopia fiction')

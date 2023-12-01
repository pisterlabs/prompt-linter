import os
import openai
import pandas as pd
from ast import literal_eval
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from dotenv import load_dotenv

# Load the environment variables from the .env file
load_dotenv()

# Define the constants for OpenAI, Qdrant, and the embedding model
EMBEDDING_MODEL = "text-embedding-ada-002"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

def get_openai_api_key():
    """Return the OpenAI API key."""
    return OPENAI_API_KEY

def initialize_qdrant_client():
    """Initialize and return the Qdrant client."""
    url = QDRANT_URL
    api_key = QDRANT_API_KEY
    client = QdrantClient(url=url, api_key=api_key)
    return client

def read_embeddings_from_csv(file_path):
    """
    Read embeddings data from a CSV file.

    Parameters:
        file_path (str): The file path of the CSV file containing embeddings data.

    Returns:
        pd.DataFrame: DataFrame containing the embeddings data.
    """
    embeddings_df = pd.read_csv(file_path)
    embeddings_df['name_vector'] = embeddings_df['name_vector'].apply(literal_eval)
    embeddings_df['description_vector'] = embeddings_df['description_vector'].apply(literal_eval)
    return embeddings_df

def create_embeddings(query):
    """
    Create embeddings for the input query using the OpenAI API.

    Parameters:
        query (str or list): The query or a list of queries to be embedded.

    Returns:
        list or ndarray: The embeddings for the input query/queries.
    """
    if isinstance(query, str):
        query = [query]

    openai.api_key = get_openai_api_key()
    embeddings = openai.Embedding.create(
        input=query,
        model=EMBEDDING_MODEL,
    )

    if len(embeddings['data']) == 1:
        return embeddings['data'][0]['embedding']
    else:
        return [entry['embedding'] for entry in embeddings['data']]

def create_collection(client, collection_name, vector_size):
    """
    Create a collection in Qdrant with the specified vector configuration.

    Parameters:
        client (QdrantClient): The initialized Qdrant client.
        collection_name (str): The name of the collection to be created.
        vector_size (int): The size of the vectors in the collection.
    """
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config={
            "name_vector": rest.VectorParams(distance=rest.Distance.COSINE, size=vector_size),
            "description_vector": rest.VectorParams(distance=rest.Distance.COSINE, size=vector_size),
        }
    )

def insert_embeddings_into_collection(client, collection_name, embeddings_df):
    """
    Insert embeddings data into the specified Qdrant collection.

    Parameters:
        client (QdrantClient): The initialized Qdrant client.
        collection_name (str): The name of the collection to insert data into.
        embeddings_df (pd.DataFrame): DataFrame containing the embeddings data.
    """
    points_to_upsert = []

    for _, row in embeddings_df.iterrows():
        product_id = row['Product ID']

        # Prepare the vector data for each point
        vector_data = {
            "name_vector": row['name_vector'],
            "description_vector": row['description_vector'],
        }

        # Prepare the payload data for each point (optional)
        payload_data = {
            "name": row["Product Name"],
            "content": row["Description"],
            "metadata": {
                "product_id": row["Product ID"],
                "product_name": row["Product Name"],
                "product_brand": row["Brand"],
            }
        }

        # Create a PointStruct object for each row and add it to the list
        point = rest.PointStruct(id=product_id, vector=vector_data, payload=payload_data)
        points_to_upsert.append(point)

    # Perform the upsert operation with the prepared list of points
    client.upsert(collection_name=collection_name, points=points_to_upsert)

def get_num_products(client, collection_name):
    """
    Get the number of products in the specified Qdrant collection.

    Parameters:
        client (QdrantClient): The initialized Qdrant client.
        collection_name (str): The name of the collection.

    Returns:
        int: The number of products in the collection.
    """
    count_result = client.count(collection_name=collection_name)
    num_products = count_result.count
    return num_products

def read_products_data(filename):
    """
    Read product data from a CSV file into a DataFrame.

    Parameters:
        filename (str): The file path of the CSV file containing product data.

    Returns:
        pd.DataFrame: DataFrame containing the product data.
    """
    products_df = pd.read_csv(filename)
    return products_df

def query_qdrant(query, collection_name, vector_name='description_vector', top_k=5):
    """
    Execute a search query using Qdrant and retrieve the top-k results.

    Parameters:
        query (str): The query string for the search.
        collection_name (str): The name of the collection to search in.
        vector_name (str): The name of the vector field to use for the search.
        top_k (int): The number of top results to retrieve.

    Returns:
        dict: The search results containing the top-k data points.
    """
    client = initialize_qdrant_client()
    embedded_query = create_embeddings(query)

    query_results = client.search(
        collection_name=collection_name,
        query_vector=(vector_name, embedded_query),
        limit=top_k,
    )

    return query_results

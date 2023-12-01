import os
import streamlit as st
import pandas as pd
import cohere
import pinecone
from tqdm.auto import tqdm

project_ideas = [
    "To-Do List Application",
    "Weather App",
    "URL Shortener",
    "Image Gallery",
    "Chatbot",
    "News Aggregator",
    "Password Generator",
    "Expense Tracker",
    "Quiz Game",
    "Social Media Dashboard",
    "File Encryption/Decryption",
    "Language Translator",
    "Stock Market Analyzer",
    "Music Player",
    "Recipe Finder"
]
project_list = [{'id': id+1, 'idea': idea} for id, idea in enumerate(project_ideas)]

dataset = pd.DataFrame(project_list)

cohere_api_key = os.environ['COHERE_API_KEY']

def encode_text(text):
    """
    Encodes the given text using the Cohere API.
    Args:
        text (str): The text to encode.
    Returns:
        numpy.ndarray: The embeddings of the given text.
    """

    # Create a CohereClient instance with the given API key.
    cohere_client = cohere.Client(api_key=cohere_api_key)

    # Send the text to the Cohere API for encoding.
    response = cohere_client.embed(texts=[text], 
                                   model='large', 
                                   truncate='LEFT')

    # Return the embeddings of the text.
    return response.embeddings[0]

def create_index(index_name):
    """
    Creates a Pinecone index with the given name if it doesn't already exist and returns the index.
    Args:
        index_name (str): The name of the index to create.
    Returns:
        pinecone.Index: The newly created or existing index.
    """
    # Initialize Pinecone with API key and GCP environment.
    pinecone.init(
        api_key=os.environ['PINECONE_API_KEY'],
        environment='asia-northeast1-gcp'
    )

    # Create the index if it doesn't already exist.
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            name=index_name,
            dimension=4096,
            metric='cosine',
            pods=1, 
            pod_type="s1.x1"
        )

    # Return the index.
    return pinecone.Index(index_name)


def index_questions(questions, index, batch_size: int = 5) -> None:
    """
    Indexes a list of questions in an Annoy index.
    Args:
        questions: A list of strings, where each string represents a question to be indexed.
        index: An instance of the AnnoyIndex class that represents the index to be used.
        batch_size: An integer that represents the number of questions to index at a time. Defaults to 128.
    """
    # Iterate over the questions in batches.
    for i in tqdm(range(0, len(questions), batch_size)):
        # Determine the end index of the current batch.
        i_end = min(i+batch_size, len(questions))
        # Create a list of IDs for the current batch.
        ids = [str(x) for x in range(i, i_end)]
        # Create a list of metadata objects for the current batch.
        metadatas = [{'text': text} for text in questions[i:i_end]]
        # Create a list of embeddings for the current batch.
        embeddings = [encode_text(text) for text in questions[i:i_end]]
        # Zip the IDs, embeddings, and metadata objects together into a list of records.
        records = zip(ids, embeddings, metadatas)
        # Upsert the records into the index.
        index.upsert(records)

def query_index(index, query, top_k=3):
    """
    Queries an index for the top-k most relevant results to a given query.
    Args:
        index (pinecone.Index): The index to query.
        query (str): The query string to search for.
        top_k (int): The number of results to return. Defaults to 3.
    Returns:
        results (dict): A dictionary containing the top-k most relevant results.
    """
    # Encode the query string.
    xq = encode_text(query)

    # Query the index for the top-k most relevant results, including metadata.
    results = index.query(xq, top_k=top_k, include_metadata=True)

    # Return the top-k most relevant results.
    return results


def delete_index(index_name):
    """
    Deletes the Pinecone index with the given name.
    Args:
        index_name (str): The name of the index to delete.
    """
    # Initialize the Pinecone API with the API key stored in the environment variable.
    pinecone.init(api_key=os.environ['PINECONE_API_KEY'])

    # Delete the index with the given name.
    pinecone.delete_index(index_name)


def main():
    """
    This is the main function that runs the semantic search application.
    """

    # Set the title of the Streamlit app.
    st.title("Semantic Search Application")

    # Load the dataset and extract questions.
    # dataset = load_dataset('quora', split='train')
   # Load the project ideas from the DataFrame.
    df = pd.DataFrame({
        'id': [i for i in range(1, len(project_ideas) + 1)],
        'idea': project_ideas
    })

    # Extract the questions from the DataFrame.
    questions = df['idea'].tolist()

    # Create and index the questions.
    index_name = 'semantic-search-fast'
    index = create_index(index_name)
    index_questions(questions, index)

    # Get user query and display search results.
    query = st.text_input("Enter your query:")
    if st.button("Search"):
        results = query_index(index, query)
        st.write("Top search results:")
        for result in results['matches']:
            st.write(f"{round(result['score'], 2)}: {result['metadata']['text']}")

    # Delete the index.
    delete_index(index_name)


if __name__ == "__main__":
    main()

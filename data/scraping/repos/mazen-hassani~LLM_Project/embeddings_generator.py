import pandas as pd
from langchain.embeddings import CohereEmbeddings
import os
from utils import save_to_vector_database
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

mental_health_faq_filename = os.getenv("FAQ_DB")
df = pd.read_csv(mental_health_faq_filename, nrows=10)

cohere_api_key = os.getenv("COHERE_API_KEY")
embeddings = CohereEmbeddings(cohere_api_key=cohere_api_key)
embeddings_vectors = []

for index, row in df.iterrows():
    # Extract the question text from the current row
    question = row['Questions']
    
    # Generate an embedding for the question using the Cohere API
    embedding = embeddings.embed_query(question)    
    # Store the embedding in a dictionary with the question ID as key
    embeddings_vectors.append(embedding)
# Save the embeddings to a vector database (e.g., Elasticsearch or Faiss)
save_to_vector_database(embeddings_vectors)

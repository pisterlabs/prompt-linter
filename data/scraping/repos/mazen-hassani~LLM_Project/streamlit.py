import streamlit as st
from langchain.embeddings import CohereEmbeddings
import os
from utils import find_closest_embedding
import pandas as pd
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

cohere_api_key = os.getenv("COHERE_API_KEY")
mental_health_faq_filename = os.getenv("FAQ_DB")

df = pd.read_csv(mental_health_faq_filename)
embeddings = CohereEmbeddings(cohere_api_key=cohere_api_key)

st.title("Mental Health FAQ")

# Add a text input widget for the user to enter their question
prompt = st.text_input("Enter your question about mental health:")

# Add a button widget to trigger the search
if st.button("Search"):
    # Generate an embedding for the question using the Cohere API
    embedding = embeddings.embed_query(prompt)

    index = find_closest_embedding(embedding)

    # Add a text output widget to display the answer
    st.write(df.iloc[index[0][0]]["Answers"])
# Embedding Support

import umap
import pandas as pd
import numpy as np
import streamlit as st

from src.constants import OPENAI_KEY
# from langchain.embeddings import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer

# def generate_embeddings_openai(txt_series):
#     embedder = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)
#     na_filled = txt_series.fillna("", inplace=False) 
#     # Generate embeddings for the text column
#     return embedder.embed_documents(na_filled.tolist())


def generate_embeddings_free(txt_series):
    embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    na_filled = txt_series.fillna("", inplace=False) 
    # Generate embeddings for the text column
    return embedder.encode(na_filled.tolist())
    

@st.cache_data(show_spinner=False)
def embed_reviews(df, column):
    df[f'{column}_embeddings'] = pd.Series(list(generate_embeddings_free(df[column])))
   
    return df

@st.cache_data(show_spinner=False)
def reduce_dimensions_append_x_y(df, vector_col):
    df = df.copy()
    
    # Extract embeddings, cluster labels, and hover text from DataFrame
    embeddings = np.array(df[vector_col].tolist())
    # Apply UMAP
    reducer = umap.UMAP(random_state=42)
    embeddings_2d = reducer.fit_transform(embeddings)
    df['x'] = embeddings_2d[:, 0]
    df['y'] = embeddings_2d[:, 1]

    return df
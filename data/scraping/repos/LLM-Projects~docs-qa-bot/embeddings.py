# Create embeddings of the extracted text for futhrer preprocessing

## Steps involved are:
## - Create chunks of the text so that we can hadle bulk data if any
## - Then create documents of every chunk
## - Finally create embeddings using the Hugging Face embedding class

import pandas as pd
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
import streamlit as st

@st.cache_data
def create_embeddings(text, OPENAI_API_KEY):
    text_splitter = CharacterTextSplitter(        
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap  = 200,
        length_function = len,
    )

    texts = text_splitter.create_documents([text])

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    embeddings_query_result = []

    for i in range(len(texts)):
        query_result = embeddings.embed_query(texts[i].page_content)
        embeddings_query_result = embeddings_query_result + query_result

    embeddings_df = pd.DataFrame(embeddings_query_result)
    embeddings_df.columns = ["embedded_values"]
    
    return texts, embeddings, embeddings_df

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 12:12:41 2023

@author: amanmujeeb
"""

import streamlit as st
from langchain.llms import OpenAI
from langchain.agents import create_pandas_dataframe_agent
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
import pandas as pd

def generate_response(uploaded_file, openai_api_key, query_text):
    # Load document if file is uploaded
    if uploaded_file is not None:
        #documents = [uploaded_file.read().decode()]
        # Split documents into chunks
        ##text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        ##texts = text_splitter.create_documents(documents)
        # Select embeddings
        ##embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        # Create a vectorstore from documents
        ##db = Chroma.from_documents(texts, embeddings)
        # Create retriever interface
        ##retriever = db.as_retriever()
        # Create QA chain
        #qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=openai_api_key), chain_type='stuff', retriever=retriever)
        df = pd.read_csv(uploaded_file)
        qa = create_pandas_dataframe_agent(OpenAI(openai_api_key=openai_api_key), 
                                         df, 
                                         verbose=True)
        return qa.run(query_text)

# Page title
st.set_page_config(page_title='Ask the Csv App')
st.title('Ask the Csv App')

# File upload
uploaded_file = st.file_uploader('Upload a Csv', type='csv')
#logo
logo = "andromeda.jpeg"  # Replace with the actual filename of your logo
st.sidebar.image(logo, use_column_width=True)
# Query text
query_text = st.text_input('Enter your question:', placeholder = 'Please provide a short summary.', disabled=not uploaded_file)

# Form input and query
result = []

with st.form('myform', clear_on_submit=True):
    openai_api_key = st.text_input('OpenAI API Key', type='password', disabled=not (uploaded_file and query_text))
    submitted = st.form_submit_button('Submit', disabled=not(uploaded_file and query_text))
    if submitted and openai_api_key.startswith('sk-'):
        with st.spinner('Calculating...'):
            response = generate_response(uploaded_file, openai_api_key, query_text)
            result.append(response)
            
            del openai_api_key

if len(result):
    st.info(response)

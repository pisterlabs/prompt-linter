#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 23:07:34 2023

@author: shbmsk
"""
import os
import streamlit as st
import tempfile
import sqlite3
import streamlit as st
#from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain import SQLDatabase, SQLDatabaseChain
from apiKey import openapi_key , serpapi_key


os.environ['OPENAI_API_KEY'] = openapi_key
# Storing the response
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

def generate_response(message):
    # Connect to the database
    dburi = "sqlite:///chinook.db"
    db = SQLDatabase.from_uri(dburi)

    # Create an instance of LLM
    llmBista = ChatOpenAI()

    # Create an SQLDatabaseChain using the ChatOpenAI model and the database
    db_chain = SQLDatabaseChain.from_llm(llm=llmBista, db=db)
    ai_response = db_chain.run(message)

    return ai_response

def get_text():
    # Get user input from text input field
    input_text = st.text_input("You: ", "", key="input")
    return input_text  

def main():
    # Load environment variables
    os.environ['OPENAI_API_KEY'] = openapi_key

    # Display header
    st.header('Bista Query')

    # Get user input
    user_input = get_text()

    if user_input:
        # Generate response for the user input
        st.session_state["generated"] = generate_response(user_input)

    if st.session_state['generated']:
        # Display the generated response
        st.write(st.session_state['generated'])

if __name__ == '__main__':
    main()
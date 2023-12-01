import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
import psycopg2
import requests
import langchain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# from langchain.llms import GPT4All, LlamaCpp
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import OpenAI
import openai
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain

import os
import sqlite3
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

API_URL = 'http://127.0.0.1:8000/'



def get_collections():
    # Path to the SQLite database file
    db_path = "db/chroma.sqlite3"

    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Execute the query to fetch the 'name' column from the 'collections' table
    cursor.execute("SELECT name FROM collections")

    # Fetch all the results and extract the names into a list
    names = [row[0] for row in cursor.fetchall()]
    # Close the connection
    conn.close()

    return names


def connect_to_db(host, port, user, password, dbname):
    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            dbname=dbname
        )
        return conn
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None


def process_query(query, collection):
    url = f"{API_URL}/api/chat/"
    response = requests.post(
        url, json={"query": query, "collection_name": collection})
    return response.json().get('answers', '')


def main():
    st.markdown("""
    <style>
    /* change user bubble color */
    .stChat .stChat__bubble--user {
        background-color: lightblue !important;
    }
    /* change assistant bubble color */
    .stChat .stChat__bubble--bot {
        background-color: lightgreen !important;
    }
    </style>
    """, unsafe_allow_html=True)
    # Database setup
    input_db = SQLDatabase.from_uri(
        "postgresql://shaund:thembisile@localhost:5432/postgres")
    llm_1 = OpenAI(temperature=0.7, verbose=True)

    # Create SQLDatabaseChain
    db_agent = SQLDatabaseChain.from_llm(llm=llm_1,
                                db=input_db,
                                verbose=True)
    st.markdown("<h1 style='text-align: center; color: black;'>💬 HPCSA Agent</h1>",
                unsafe_allow_html=True)

    # Sidebar for "DocumentUploading"
    with st.sidebar:
        st.header("Upload Documents")
        uploaded_file = st.file_uploader("Choose a file to upload", type=[
                                         "txt", "pdf", "docx", "csv"], accept_multiple_files=False)

        collections = get_collections()
        selected_collection = st.selectbox('Select Space or Add New:', ['Select Space'] + collections + ['Add New Space'], key="selected_collection")


        if selected_collection == 'Add New Space':
            collection_name = st.text_input("New Space Name:")
        else:
            collection_name = selected_collection

        if st.button("Upload") and uploaded_file and collection_name:
            embed_documents(uploaded_file, collection_name)

    colored_header(label='', description='', color_name='blue-30')

    # st.header("ChatUI")
    # Assistant initiate response in session
    if 'generated' not in st.session_state:
        st.session_state['generated'] = [
            "Hi, I am a virtual assistant for HPCSA, How may I help you?"]

    # user question
    if 'user' not in st.session_state:
        st.session_state['user'] = ['Hi!']

    response_container = st.container()
    input_container = st.container()

    # Clear input text
    st.session_state['input_text'] = ''

    # Applying the user input box
    with input_container:
        user_input = st.chat_input("Your message: ")
        is_sql_query = st.checkbox('Is this a report related query?')

        if not is_sql_query:
            collections = ['Select Space'] + collections
            selected_collection = st.selectbox(
                'Select Space:', collections, key="query_collection_name",
                index=0 if 'query_collection_name' not in st.session_state else collections.index(st.session_state.query_collection_name))
            # if st.button("Refresh Collections"):
            #     collections = get_collections() 
            # Update session state only if the selected collection has changed
            if 'query_collection_name' not in st.session_state or selected_collection != st.session_state.query_collection_name:
                st.session_state.query_collection_name = selected_collection

        if user_input:
            # Check if the input is a SQL query
            if is_sql_query:
                # Pass the user's query to the Langchain agent
                with st.spinner('Processing your query...'):
                    try:
                        answer = db_agent.run(user_input)
                        print(answer)
                        # Append the answer directly from the SQLDatabaseChain
                        st.session_state.generated.append(answer)
                        st.session_state.user.append(user_input)
                    except ValueError as e:
                        if "Requested tokens" in str(e) and "exceed context window" in str(e):
                            st.error(
                                "The query is too long! Please shorten it and try again.")
                        else:
                            raise
            else:
                # Check if the query_collection_name is selected
                if st.session_state.query_collection_name == 'Select Space':
                    st.warning("Please select a space before submitting.")
                else:
                    # Go ahead and process the query
                    with st.spinner('Processing your query...'):
                        try:
                            res = process_query(
                                user_input, st.session_state.query_collection_name)
                            # Append the answer from the local model
                            st.session_state.generated.append(res)
                            st.session_state.user.append(user_input)
                        except Exception as e:
                            st.write(f"An error occurred: {e}")

    with response_container:
        if st.session_state['generated']:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state['user'][i], is_user=True, key=str(
                    i) + '_user', avatar_style="avataaars", seed="SD")
                message(st.session_state["generated"][i], key=str(i))
                


def embed_documents(uploaded_file, collection_name):
    url = f"{API_URL}/api/upload/"
    files_to_send = {
        'file': (uploaded_file.name, uploaded_file.read(), uploaded_file.type)}
    response = requests.post(url, files=files_to_send, data={
                             'collection_name': collection_name})
    result = response.json()

    if response.status_code == 200:
        st.success(result['message'])
        st.write("Saved Files:", result['saved_files'])
    else:
        st.error(f"Error: {result}")

    return result


if __name__ == "__main__":
    main()

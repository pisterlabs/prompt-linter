import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from langchain import OpenAI, SQLDatabase, SQLDatabaseChain
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Streamlit app starts here
st.title("Analyse your data")

# Upload CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    # Read the CSV file using pandas
    df = pd.read_csv(uploaded_file)

    # Create a temporary SQLite database file path
    temp_db_file_path = "temp_data.sqlite"

    # Create a connection to the temporary SQLite database
    engine = create_engine(f"sqlite:///{temp_db_file_path}")

    # Convert the DataFrame to a temporary SQLite database table
    # Replace 'table_name' with the desired table name
    df.to_sql("data", engine, if_exists="replace", index=False)

    # Once the data is inserted, the temporary SQLite database is created with the given data

    # Load Langchain components
    dburi = f"sqlite:///{temp_db_file_path}"
    db = SQLDatabase.from_uri(dburi)
    llm = OpenAI(temperature=0)
    db_chain = SQLDatabaseChain(llm=llm, database=db, verbose=True)

    # Define functions for Langchain queries
    def run_langchain_query(query):
        response = db_chain.run(query)
        return response

    # User input for Langchain queries
    user_input = st.text_input("Enter your question:", "What is the most popular car in the dataset?")

    if st.button("Run Query"):
        response = run_langchain_query(user_input)
        st.write(f"Query: {user_input}")
        st.write(f"Response: {response}")

    # Close the connection to the temporary SQLite database
    engine.dispose()

    

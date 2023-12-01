import streamlit as st
import sqlite3
import tempfile
import os
from langchain.chains import SQLDatabaseSequentialChain
from langchain import OpenAI, SQLDatabase, SQLDatabaseChain
from langchain.prompts.prompt import PromptTemplate

os.environ['OPENAI_API_KEY'] = 'sk-***********************************************'

# Function to save file contents to a temporary file
def save_file_to_temp(file):
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file_path = temp_file.name

    # Save the file contents to the temporary file
    with open(temp_file_path, 'wb') as f:
        f.write(file.read())
    return temp_file_path


# Streamlit app code
def main():
    # Display file uploader widget
    st.header("Query your database using Natural Language.")
    file = st.file_uploader("Upload a database file", type=["db"])

    if file is not None:
        # Save the file to a temporary file
        temp_file_path = save_file_to_temp(file)

        # Initialize SQLDatabaseSequentialChain from the temporary file
        db = SQLDatabase.from_uri(f"sqlite:///{temp_file_path}")


        # Perform operations on the database
        try:
            llm = OpenAI(temperature=0, verbose=True)
            chain = SQLDatabaseSequentialChain.from_llm(llm, db, verbose=True, return_intermediate_steps=True)
            question = st.text_input("Enter your query about the database.")
            if question:
                st.write(chain(question))
            else:
                print()

        except:
            print()


# Run the Streamlit app
if __name__ == "__main__":
    main()

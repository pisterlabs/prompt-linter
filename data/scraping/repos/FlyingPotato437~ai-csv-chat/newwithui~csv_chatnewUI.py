from langchain.agents import create_csv_agent
from langchain.llms import OpenAI as LangchainOpenAI
import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import tempfile

def main():
    load_dotenv()
    
    # Set the page layout
    st.set_page_config(layout='wide', page_title="CSVChat powered by OpenAI and created by Srikanth Samy 🤖")
    
    # Title of the application
    st.title("CSVChat powered by OpenAI 🤖")
    st.subheader("Created by Srikanth Samy")

    # File uploader for CSV files
    input_csv = st.file_uploader("Upload your CSV file", type=['csv'])

    # If a CSV file is uploaded
    if input_csv is not None:
        # Create a temporary file and write the contents of the uploaded file into it
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            temp_file.write(input_csv.getvalue())
            temp_path = temp_file.name

        # Create two columns
        col1, col2 = st.columns([1,1])

        # First column: display CSV data
        with col1:
            st.info("CSV Uploaded Successfully")
            data = pd.read_csv(input_csv)
            st.dataframe(data, use_container_width=True)

        # Second column: chat interface
        with col2:
            st.info("Chat Below")
            user_question = st.text_area("Enter your query")

            # If a query is entered
            if user_question is not None and user_question != "":
                if st.button("Chat with CSV"):
                    st.info("Your Query: " + user_question)

                    # Create an agent using Langchain
                    agent = create_csv_agent(
                        LangchainOpenAI(temperature=1), temp_path, verbose=True)

                    # Process the query and display the result
                    with st.spinner(text="In progress..."):
                        st.success(agent.run(user_question))

if __name__ == "__main__":
    main()

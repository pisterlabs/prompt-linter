## Using OpenAi's GPT-3 to answer questions about a CSV file

from langchain.agents import create_csv_agent
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os
import streamlit as st
import tempfile

def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your CSV")
    st.header("Ask your CSV 📈")

    csv_file = st.file_uploader("Upload a CSV file", type="csv")
    if csv_file is not None:
        llm = OpenAI(temperature=0.9)
        
        # Create a temporary file to store the uploaded CSV data
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            temp_file.write(csv_file.read())
            temp_file_path = temp_file.name
        
        agent = create_csv_agent(llm, temp_file_path, verbose=True)

        user_question = st.text_input("Ask a question about your CSV: ")

        if user_question is not None and user_question != "":
            with st.spinner(text="In progress..."):
                st.write(agent.run(user_question))

if __name__ == "__main__":
    main()






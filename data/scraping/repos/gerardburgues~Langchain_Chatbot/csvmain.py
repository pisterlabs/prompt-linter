from langchain.agents import create_csv_agent
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os
import streamlit as st

from dotenv import load_dotenv

import pandas as pd

# Create a List of Documents from all of our files in the ./docs folder
load_dotenv()

openai_api_key = os.getenv("OPENAI_KEY")


def main():
    load_dotenv()

    # Load the OpenAI API key from the environment variable

    st.set_page_config(page_title="Ask your CSV")
    st.header("Ask your CSV 📈")

    csv_file = st.file_uploader("Upload a CSV file", type="csv")
    if csv_file is not None:
        agent = create_csv_agent(
            OpenAI(
                temperature=0,
                openai_api_key=openai_api_key,
            ),
            csv_file,
            verbose=True,
        )

        user_question = st.text_input("Ask a question about your CSV: ")

        if user_question is not None and user_question != "":
            with st.spinner(text="In progress..."):
                st.write(agent.run(user_question))


if __name__ == "__main__":
    main()

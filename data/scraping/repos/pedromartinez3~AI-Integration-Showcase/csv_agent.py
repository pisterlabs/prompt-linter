"""
Converse with your CSV file through a local web app

Usage in command prompt: 
streamlit run embed.py your_directory

Arguments:
    your_directory: Directory to embed
"""

import os
import streamlit as st
import tempfile
from dotenv import load_dotenv

from langchain.agents import create_csv_agent
from langchain.llms import OpenAI

# Load environment variables from .env file
load_dotenv()

def main():
    
    # Load API Key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set your OPENAI_API_KEY in the .env file")
    
    # set up web app
    st.set_page_config(page_title='Ask your CSV', page_icon='📊', layout='wide')
    st.header('📊 Ask your CSV')

    # get user csv file
    user_csv = st.file_uploader('Upload your CSV file here', type='csv')

    # create temp file to feed into agent
    if user_csv is not None:
        temporary_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")

        # try/except so we delete temp file even if something goes wrong    
        try:
            temporary_file.write(user_csv.getvalue())
            temporary_file.close()

            user_question = st.text_input('Converse with your CSV here:')

            if user_question:

                # create agent
                llm = OpenAI(api_key=api_key, temperature=0)
                agent = create_csv_agent(llm=llm, path=temporary_file.name, verbose=True)
                response = agent.run(user_question)
                st.write(response)
        finally:
            os.unlink(temporary_file.name)

if __name__ == '__main__':
    main()
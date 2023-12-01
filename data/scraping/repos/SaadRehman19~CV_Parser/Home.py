import streamlit as st
from Embedding import ResumeParser
resume_parser = ResumeParser()
import openai
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("OPEN_API_KEY")


def main():
     
    st.sidebar.success("Downloads")
     
    st.title("SAVYOUR - CV Search")

    query = st.text_input("Enter your query")

    search_button = st.button("Search")

    if search_button:
        perform_search(query)

def perform_search(query):
    res=openai.Embedding.create(
    input=query,
    engine="text-embedding-ada-002"
)
    embed = res.data[0].embedding
    resume_parser.search(embed,query)

if __name__ == '__main__':
    main()

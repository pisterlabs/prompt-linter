import streamlit as st
from backend import comp_press

import os
from langchain.llms import OpenAI
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain

def frontend():
    st.set_page_config(page_title="Youtbe_GPT", layout="wide")
    st.title("Chat with Your PDF")

    question = st.text_input("Ask Question below", placeholder='Your Question')

    with st.sidebar:
        st.subheader("Enter your API key")
        api_key = st.text_input("Enter API key", placeholder="sk_test_BQokikJOvBiI2HlWgH4olfQ2", type="password")
        #help="How to get an OpenAI API Key: https://www.howtogeek.com/885918/how-to-get-an-openai-api-key/"
        url = st.text_input("Enter URL below: ", placeholder="www.youtube.com/...")

    if url and api_key is not None:
        if question:
            ans = comp_press(api_key=api_key, url=url, question=question)
            st.write(ans)

if __name__ == "__main__":
    frontend()
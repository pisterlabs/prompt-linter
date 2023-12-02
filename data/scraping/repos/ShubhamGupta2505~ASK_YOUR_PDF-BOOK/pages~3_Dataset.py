import streamlit as st
# from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os
from langchain.callbacks import get_openai_callback



def main():
    st.title("CSV ")


    
if __name__ == 'main':
    main()
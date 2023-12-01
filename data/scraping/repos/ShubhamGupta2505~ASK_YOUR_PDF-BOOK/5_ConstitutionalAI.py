import streamlit as st
# from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os
from langchain.callbacks import get_openai_callback
st.set_page_config(page_title="Constitutional AI")
def main():
    st.set_page_config(page_title="Constitutional AI")
    st.title("Constitutional AI")
    st.header("Constitutional AI")
    load_dotenv()
    user_question = st.text_input("Ask a Question ")
    llm = OpenAI(model_name='text-davinci-003', 
             temperature=0.9, 
             max_tokens = 256,
             verbose=True)

if __name__ == 'main':
    main()
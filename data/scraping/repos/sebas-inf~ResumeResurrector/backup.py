#Main File
import streamlit as st
import openai
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from streamlit_extras import add_vertical_space as avs
from langchain.callbacks import get_openai_callback
import base64
import os

import pandas as pd
from tqdm import tqdm
from langchain.text_splitter import CharacterTextSplitter


st.set_page_config(page_title="Resume Reviewer", page_icon="📖")


with st.sidebar:
    st.title("Resume Reviewer")
    st.markdown('''
    ## About
    This app is a LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
                

 
    ''')
    st.write('Made by Spanish Indian Inquision')


load_dotenv()
def main():
    #openai.api_key = st.secrets["OPENAI_API_KEY"]
    openai.api_key = os.getenv("OPENAI_API_KEY")
    pdf = st.file_uploader("Upload a file", type='pdf')
    

    #st.write(pdf) #used for printing file name
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function = len
            )
        
        chunks = text_splitter.split_text(text=text)

        #st.write(chunks)
        # # embeddings
        store_name = pdf.name[:-4]
        # st.write(f'{store_name}')
        # st.write(chunks)
 
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
            # st.write('Embeddings Loaded from the Disk')s
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)



    # Input User Questions
    query = st.text_input("Ask questions about the selected book below: ")

    if query:
        docs = VectorStore.similarity_search(query=query, k=3)
        
        llm = ChatOpenAI(model_name='gpt-3.5-turbo')
        chain = load_qa_chain(llm = llm, chain_type = "stuff")
        with get_openai_callback() as cb:
            response = chain.run(input_documents = docs, question = query)
            print(cb)
        st.write(response)  

    

if __name__ == "__main__":
    main()




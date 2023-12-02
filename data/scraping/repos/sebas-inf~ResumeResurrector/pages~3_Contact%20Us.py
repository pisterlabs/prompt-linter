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
import time

import pandas as pd
from tqdm import tqdm
from langchain.text_splitter import CharacterTextSplitter
from PIL import Image
from streamlit_lottie import st_lottie
import requests

st.set_page_config(page_title="Resume Resurrector", page_icon="📖")

with st.spinner("Loading..."):
        time.sleep(1)

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code !=200:
        return None
    return r.json()

st.header("Meet the Team!")

image =Image.open('./images/teampic.jpg') 
st.image(image, caption="The Team")

with st.sidebar:
    st.image("./images/panda_logo.png", caption="")
    st.title("Resume Resurrector")
    st.markdown('''
    This app is a LLM-powered resume reviewer built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
                
    ''')


left_column, right_column = st.columns(2)
with left_column:
     st.subheader("Connect with the Team")
     st.markdown('''[Johan John Joji](https://www.linkedin.com/in/johanjjoji/)''')
     st.markdown('''[Sebastian Infante](https://www.linkedin.com/in/infante-seb/)''')
     st.markdown('''[Ijlal Sultan](https://www.linkedin.com/in/sultan-rajaijlal/)''')
     st.markdown('''[Mihir Karnani](https://www.linkedin.com/in/mihir-karnani-16x/)''')
     st.markdown('''[Rachit Kasangra](https://www.linkedin.com/in/rachit-kansagra-3424a9262/)''')
     
with right_column:
    lottie_coding = load_lottieurl("https://lottie.host/71a49c0d-c96c-41c6-afb7-434adbd8b01c/AbDotMP4M9.json")
    st_lottie(lottie_coding,height =300,key="paperplane")





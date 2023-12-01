

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


import requests
import streamlit as st 
from streamlit_lottie import st_lottie

with st.container():
    left_column,right_column = st.columns(2)

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code !=200:
        return None
    return r.json()
    
lottie_coding = load_lottieurl("https://lottie.host/e6ce3de4-1da6-4d60-84d2-100f97d68b37/otWBBaQhxU.json")



with st.container():
     st.header("Resume     Resurrector")
     st_lottie(lottie_coding,height =175,key="paperplane")
     st.divider()
     st.subheader("The Idea")
     st.write("Welcome to the future of resume optimization! Our Free Resume Checker is your passport to career excellence, meticulously scanning and fine-tuning your resume to match ATS standards. Say goodbye to missed opportunities "+
                 "and hello to precision in crafting your professional narrative. Join us on this journey towards career success!")
     st.subheader("The Creation")
     st.write("Our web app, built with Python and Streamlit for the frontend, leverages OpenAI's Language Model (LLM) integrated through Langchain for precise resume evaluation. Users input resumes and the LLM, with its natural language processing offers accurate assessments. This fusion of Python, Streamlit, Langchain and OpenAI's LLM streamlines the review process transforming it from manual to efficient.")

     st.write(" --------------------------------"+ 
              "------------------")
     st.subheader("                The Technology       ")

    
    

st.image("./images/Streamlitnew.jpg", caption="")
st.image("./images/openai2.jpg", caption="")
st.image("./images/langchain1.png", caption="")
st.image("./images/python1.jpg", caption="")







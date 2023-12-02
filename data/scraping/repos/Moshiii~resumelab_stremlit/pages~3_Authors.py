import streamlit as st
import PyPDF2
import os
import io
import time
from langchain import PromptTemplate, OpenAI, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.chains import create_extraction_chain
import fitz
test = False

st.markdown("Coding: Moshi, Carrie")
st.markdown("Prompting: Moshi, Carrie")

st.markdown("Special Thanks to Leona for UX Consulting")

st.markdown("Moshi: wmswms938@gmail.com")
st.markdown("Carrie: yanran2012@gmail.com")
st.markdown("Leona: leona.huang.toronto@gmail.com")

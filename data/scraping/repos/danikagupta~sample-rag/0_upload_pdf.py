# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any

import numpy as np

import streamlit as st
from streamlit.hello.utils import show_code



import os
import streamlit as st
from streamlit.logger import get_logger
import PyPDF2
import pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import hashlib
import openai

LOGGER = get_logger(__name__)

PINECONE_API_KEY=os.environ['PINECONE_API_KEY']
PINECONE_API_ENV=os.environ['PINECONE_API_ENV']
PINECONE_INDEX_NAME=os.environ['PINECONE_INDEX_NAME']

def pdf_to_text(uploaded_file):
    pdfReader = PyPDF2.PdfReader(uploaded_file)
    count = len(pdfReader.pages)
    text=""
    for i in range(count):
        page = pdfReader.pages[i]
        text=text+page.extract_text()
    return text

def embed(text,filename):
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
    index = pinecone.Index(PINECONE_INDEX_NAME)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap  = 200,length_function = len,is_separator_regex = False)
    docs=text_splitter.create_documents([text])
    for idx,d in enumerate(docs):
        hash=hashlib.md5(d.page_content.encode('utf-8')).hexdigest()
        embedding=openai.Embedding.create(model="text-embedding-ada-002", input=d.page_content)['data'][0]['embedding']
        metadata={"hash":hash,"text":d.page_content,"index":idx,"model":"text-embedding-ada-003","docname":filename}
        index.upsert([(hash,embedding,metadata)])
    return

def run():
    st.set_page_config(
        page_title="Hello",
        page_icon="👋",
    )

#
# Direcly access Text Input    
#
st.markdown("Upload text directly")
uploaded_text = st.text_area("Enter Text","")
if st.button('Process and Upload'):
    embedding = embed(uploaded_text,"Anonymous")
#
# Accept a PDF file using Streamlit
# Upload to Pinecone
#
st.markdown("# Upload file: PDF")
uploaded_file=st.file_uploader("Upload PDF file",type="pdf")
if uploaded_file is not None:
    if st.button('Process and Upload'):
        pdf_text = pdf_to_text(uploaded_file)
        embedding = embed(pdf_text,uploaded_file.name)
    st.write("# Welcome to Streamlit! 👋")

    st.sidebar.success("Select a demo above.")

    


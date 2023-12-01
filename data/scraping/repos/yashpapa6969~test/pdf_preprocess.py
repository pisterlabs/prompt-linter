import PyPDF2
import re
import uuid
import pandas as pd
import streamlit as st
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import CharacterTextSplitter


@st.cache_data
def pdf_reader(file, chunk_size):
    loader = UnstructuredPDFLoader("my_file.pdf")
    data = loader.load()
    text_splitter = CharacterTextSplitter(
        separator="\n\n", chunk_size=chunk_size, chunk_overlap=200
    )

    docs = text_splitter.split_documents(data)
    cleaned_text = []
    for chunks in range(len(docs)):
        text = docs[chunks].page_content
        cleaned_text.append(text)

    diction = {
        "data": cleaned_text,
        "id": [str(uuid.uuid1()) for _ in range(len(cleaned_text))],
    }
    dfs = pd.DataFrame.from_dict(diction)
    return dfs

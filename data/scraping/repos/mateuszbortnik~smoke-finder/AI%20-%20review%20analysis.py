import os
import pandas as pd
import math
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA, OpenAI
from langchain.llms import OpenAIChat
from langchain.document_loaders import DirectoryLoader
import langchain
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import streamlit as st
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


st.title("GPT module (TEST)")

openai_api_key = st.text_input(
        "API Key", 
        help="Enter Open Ai API Key")

os.environ["OPENAI_API_KEY"] = openai_api_key

query = st.text_input(
"User Query", 
help="Enter a question about rviews"
,value="What users complain about?")

# read file
uploaded_file = st.file_uploader("Choose a csv file")
if st.button("Let's go"):
    # if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df)

    loader = langchain.document_loaders.DataFrameLoader(df, 'review_text')
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    docsearch = Chroma.from_documents(texts, embeddings)
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever = docsearch.as_retriever())

    
    
    # if st.button("Get answer"):
    a=st.write(qa.run(query))
    st.write(a)
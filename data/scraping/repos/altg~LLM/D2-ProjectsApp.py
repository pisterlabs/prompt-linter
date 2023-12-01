# Streamlit App to analyse Projects using GPT

import pandas as pd
import streamlit as st

import os
import openai

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

from langchain.document_loaders import DataFrameLoader

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

@st.cache_data
def load_data():

    data_loader = st.text("Loading Data ....")
    df = pd.read_csv("Results.csv")

    data_loader.text("Loading Data .... Done !")

    data_prep = st.text("Preparing Data ...")

    selected_fields = ['Scope' , 'Project Relevance' , 'Project Context' , 'Project Objective' , 'Project Rationale']

#selected_fields = [ 'Project Objective' ]

    filtered_df = df[df['Free_Text_title'].isin(selected_fields) & df['Is_latest_text'] == 1 ]

    df2 = filtered_df[['Project' , 'Free_Text_title' , 'Text']].pivot(index = ['Project'] , columns='Free_Text_title' , values = 'Text').dropna()

    df2.reset_index(inplace=True)

    data_prep.text("Preparing Data ... Done !")

    return df2[['Project' , 'Project Objective']].rename(columns={'Project' : 'source'})


@st.cache_resource
def do_embeddings(df_in):
    embeddings = OpenAIEmbeddings()

    embed = st.text("Generating Data Embeddings .....")
    loader = DataFrameLoader(df_in, page_content_column="Project Objective")

    docs = loader.load()

    store = Chroma.from_documents(docs, embeddings)

    embed.text("Generating Data Embeddings ..... Done ! ")

    return store


@st.cache_resource
def load_embeddings():
    embeddings = OpenAIEmbeddings()

    embed = st.text("Loading Data Embeddings .....")
    store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings, collection_name='projects' )

    embed.text("Loading Data Embeddings ..... Done ! ")

    return store

st.title("Project Analysis - GPT")

#df_in = load_data()

#store = do_embeddings(df_in)

store = load_embeddings()

retriever = store.as_retriever()

llm = ChatOpenAI(temperature = 0.0, model_name="gpt-3.5-turbo-16k-0613")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever, 
    verbose=True
)


response = ""
#question = st.text_input("What's you question?" )

# if question:
#      with st.spinner('Analysing...'):
#             #response = qa_chain.run(question)


if 'something' not in st.session_state:
    st.session_state.something = ''

def submit():
    st.session_state.something = st.session_state.widget
    st.session_state.widget = ''

st.text_input("What's you question?", key='widget', on_change=submit)

question = st.session_state.something

if question:
    with st.spinner('Analysing...'):
        st.write(f'Question: {question}')
        st.divider()
        response = qa_chain.run(question)
        st.write(response)


#st.write(f'Last submission: {st.session_state.something}')

# with st.form('myform', clear_on_submit=True):
    
#     submitted = st.form_submit_button('Submit' , disabled= not question)
#     if submitted:
#         with st.spinner('Calculating...'):
#             response = qa_chain.run(question)
#             #result.append(response)
            
#st.write(response)
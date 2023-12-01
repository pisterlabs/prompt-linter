# https://alphasec.io/generative-question-answering-with-langchain-and-pinecone/
import os, tempfile
import streamlit as st, pinecone
from langchain.llms.openai import OpenAI
from langchain.vectorstores.pinecone import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader

# Streamlit app
st.subheader('Generative Q&A with LangChain & Pinecone')
            
# Get OpenAI API key, Pinecone API key and environment, and source document input
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API key", os.environ.get("OPENAI_API_KEY"), type="password")
    pinecone_api_key = st.text_input("Pinecone API key", os.environ.get("TARGET_PINECONE_API_KEY"), type="password")
    pinecone_env = st.text_input("Pinecone environment", os.environ.get("TARGET_PINECONE_ENVIRONMENT"))
    pinecone_index = st.text_input("Pinecone index name", os.environ.get("TARGET_PINECONE_INDEX_NAME"))
query = st.text_input("Enter your query")

if st.button("Submit"):
    # Validate inputs
    if not openai_api_key or not pinecone_api_key or not pinecone_env or not pinecone_index or not query:
        st.warning(f"Please upload the document and provide the missing fields.")
    else:
        try:
            pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
            index = pinecone.Index(pinecone_index)
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model="text-embedding-ada-002")
            vectordb = Pinecone(index, embeddings.embed_query, "text")

            retriever = vectordb.as_retriever()

            # Initialize the OpenAI module, load and run the Retrieval Q&A chain
            llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
            qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever)
            response = qa.run(query)
            
            st.success(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
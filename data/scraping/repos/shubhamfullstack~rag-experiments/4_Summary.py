import os, tempfile
import streamlit as st
from langchain.llms.openai import OpenAI
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
from utils.authenticate import authenticate
from configs.apikey import apikey

os.environ["OPENAI_API_KEY"] = apikey  

auth  =  authenticate()
if auth[0]:
    st.subheader('Document Summary')
    source_doc = st.file_uploader("Upload Source Document", type="pdf")
    if st.button("Summarize"):
        if not source_doc:
            st.error("Please provide the source document.")
        else:
            try:
                with st.spinner('Please wait...'):
                # Save uploaded file temporarily to disk, load and split the file into pages, delete temp file
                    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                        tmp_file.write(source_doc.read())
                    loader = PyPDFLoader(tmp_file.name)
                    pages = loader.load_and_split()
                    os.remove(tmp_file.name)

                    # Create embeddings for the pages and insert into Chroma database
                    embeddings=OpenAIEmbeddings()
                    vectordb = Chroma.from_documents(pages, embeddings)

                    # Initialize the OpenAI module, load and run the summarize chain
                    llm=OpenAI(temperature=0)
                    chain = load_summarize_chain(llm, chain_type="stuff")
                    search = vectordb.similarity_search(" ")
                    print(search)
                    summary = chain.run(input_documents=search, question="Write a summary within 200 words.")

                    st.success(summary)
            except Exception as e:
                st.exception(f"An error occurred: {e}")

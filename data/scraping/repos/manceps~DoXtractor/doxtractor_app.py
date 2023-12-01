#! /bin/python3

import os
import dill
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GooglePalmEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import GooglePalm
from langchain.chains.question_answering import load_qa_chain
from langchain.agents import Tool
from langchain.tools import DuckDuckGoSearchRun

# Configure Streamlit and hide hamburger and tagline
st.set_page_config(page_title="DoXtractor | Document Knowledge Extractor", page_icon=":page_facing_up:")
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

def main():
    # st.header("Extract information from documents")

    # Define search tool
    search = DuckDuckGoSearchRun()
    tools = [
        Tool(
            name = "Current Search",
            func=search.run,
            description="useful for when you need to answer questions about current events or the current state of the world or when finding news or recent information"
        ),
    ]

    # Upload multiple PDF files
    pdf_files = st.file_uploader("Upload one or more PDF documents", type='pdf', accept_multiple_files=True)

    if pdf_files:
        try:
            with st.spinner("Processing"):
                # Process each uploaded PDF file and combine text
                for index, pdf in enumerate(pdf_files):
                    pdf_reader = PdfReader(pdf)
                    text = ""
                    
                    # Generate a unique name for the VectorStore based on the PDF filename
                    store_name = pdf.name[:-4]

                    # Extract text from PDF pages
                    for page in pdf_reader.pages:
                        text += page.extract_text()

                    # Split the extracted text into smaller chunks
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=2000,
                        chunk_overlap=100,
                        length_function=len
                    )
                    chunks = text_splitter.split_text(text=text)

                    # Check if VectorStore exists on disk, if not, create and save it
                    if os.path.exists(f"{store_name}.pkl"):
                        with open(f"{store_name}.pkl", "rb") as f:
                            VectorStore = dill.load(f)
                    else:
                        embeddings = GooglePalmEmbeddings()
                        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
                        with open(f"{store_name}.pkl", "wb") as f:
                            dill.dump(VectorStore, f)
                    
                    # Combine VectorStores from all documents
                    if index==0:
                        VectorStores = VectorStore
                    else:
                        VectorStores.merge_from(VectorStore)

            # Accept user questions/queries
            query = st.text_input("Ask questions about your documents:")

        except Exception as e:
            st.write("Something went wrong! Please remove the file and try a different document.")


        else:
            if query:
                with st.spinner("Processing"):
                    # Perform similarity search to find relevant documents
                    docs = VectorStores.similarity_search(query=query, k=3)

                    # Initialize the language model and QA chain
                    llm = GooglePalm()
                    llm.temperature = 0.1
                    chain = load_qa_chain(llm=llm, chain_type="stuff")

                    # Execute the QA chain to answer the user's query
                    response = chain.run(input_documents=docs, question=query, tools=tools)
                st.write(response)

if __name__ == '__main__':
    main()

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

import os
os.environ['OPENAI_API_KEY'] = 'YourApiKey'

def app():
    st.set_page_config(page_title='PDF App')
    st.header('PDF App')

    pdf = st.file_uploader('Upload PDF', type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()

        # st.write(text)
        split_text = RecursiveCharacterTextSplitter(
            chunk_size=2000, 
            chunk_overlap=0,
            length_function=len
        )

        texts = split_text.split_text(text)
        # st.write(texts)

        if texts is not None:
            embeddings = OpenAIEmbeddings()
            vectors = FAISS.from_texts(texts, embeddings)
 
            question = st.text_input('What is your question?')
            if question:
                docs = vectors.similarity_search(question, k=5)
                llm = OpenAI(
                    temperature=0,
                )

                chain = load_qa_chain(llm, chain_type='stuff')
                response = chain.run(input_documents=docs, question=question)
                st.write(response)


app()
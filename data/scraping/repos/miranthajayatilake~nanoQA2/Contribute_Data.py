import streamlit as st
from io import StringIO

from tqdm import tqdm
import datetime
import os
import time
import glob
from PIL import Image

from langchain.document_loaders import TextLoader

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import UnstructuredWordDocumentLoader

import pinecone



INDEX_NAME = os.environ["INDEX_NAME"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_ENV = os.environ["PINECONE_ENV"]
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

NAMESPACE = os.environ["NAMESPACE"]


image = Image.open('./utils/logo.jpg')
st.image(image, width=200)

st.title('Contribute Data')


def save_uploadedfile(uploadedfile):
    with open(os.path.join("pages/tempDir", uploadedfile.name),"wb") as f:
        f.write(uploadedfile.getbuffer())
    
    return print("Saved File:{} to pages/tempDir".format(uploadedfile.name))

def save_text(text, name):
    with open(os.path.join("pages/tempDir", name),"w") as f:
        f.write(text)
    
    return print("Saved File:{} to pages/tempDir".format(name))

def reinitate_connetion():
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    index = pinecone.Index(INDEX_NAME)


with st.expander("URL"):
    url_input = st.text_input('URL', '')

    if st.button("Read from URL"):

        with st.spinner('Wait for it...'):
            urls = [
                url_input
            ]
            loader = UnstructuredURLLoader(urls=urls)

            documents = loader.load()

            text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
            documents = text_splitter.split_documents(documents)

            embeddings = OpenAIEmbeddings()

            isNotDone = True
            while(isNotDone):
                try:
                    # st.info('1')
                    reinitate_connetion()

                    vectorstore = Pinecone.from_documents(documents, embeddings, text_key='text', index_name=INDEX_NAME, namespace=NAMESPACE)
                    isNotDone = False
                except:
                    pass

        st.info('Done')


with st.expander("PDF"):
    uploaded_pdf_files = st.file_uploader("Choose a PDF file", type=['pdf'], accept_multiple_files=True)

    if st.button("Read from PDF"):

        # st.info(uploaded_pdf_files)

        for uploaded_pdf_file in uploaded_pdf_files:

            index = pinecone.Index(INDEX_NAME)

            datafile = save_uploadedfile(uploaded_pdf_file)
            with st.spinner('Wait for it...'):

                loader = PyPDFLoader(f"pages/tempDir/{uploaded_pdf_file.name}")
                documents = loader.load()

                # st.info("Doc loaded")

                text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
                documents = text_splitter.split_documents(documents)

                # st.info("DOc splitted")

                embeddings = OpenAIEmbeddings()

                isNotDone = True
                while(isNotDone):
                    try:
                        # st.info('1')
                        reinitate_connetion()

                        vectorstore = Pinecone.from_documents(documents, embeddings, text_key='text', index_name=INDEX_NAME, namespace=NAMESPACE)
                        isNotDone = False
                    except:
                        pass

            try:
                os.remove(f"pages/tempDir/{uploaded_pdf_file.name}")
            except:
                print("file was not deleted")

            st.info(f'Done reading')

with st.expander("Word document"):
    uploaded_word_file = st.file_uploader("Choose a Word document", type=['docx'])

    if st.button("Read from Word"):

        index = pinecone.Index(INDEX_NAME)

        datafile = save_uploadedfile(uploaded_word_file)
        with st.spinner('Wait for it...'):

            loader = UnstructuredWordDocumentLoader(f"pages/tempDir/{uploaded_word_file.name}")
            documents = loader.load()

            text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
            documents = text_splitter.split_documents(documents)

            embeddings = OpenAIEmbeddings()

            isNotDone = True
            while(isNotDone):
                try:
                    # st.info('1')
                    reinitate_connetion()

                    vectorstore = Pinecone.from_documents(documents, embeddings, text_key='text', index_name=INDEX_NAME, namespace=NAMESPACE)
                    isNotDone = False
                except:
                    pass

        try:
            os.remove(f"pages/tempDir/{uploaded_word_file.name}")
        except:
            print("file was not deleted")

        st.info('Done')

with st.expander("Text"):
    pasted_text = st.text_area('Paste some text')

    if st.button("Read from text"):

        index = pinecone.Index(INDEX_NAME)

        datafile = save_text(pasted_text, "saved_text")

        with st.spinner('Wait for it...'):

            loader = TextLoader(f"pages/tempDir/saved_text")
            documents = loader.load()

            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            documents = text_splitter.split_documents(documents)

            embeddings = OpenAIEmbeddings()

            isNotDone = True
            while(isNotDone):
                try:
                    # st.info('1')
                    reinitate_connetion()

                    vectorstore = Pinecone.from_documents(documents, embeddings, text_key='text', index_name=INDEX_NAME, namespace=NAMESPACE)
                    isNotDone = False
                except:
                    pass

        try:
            os.remove("pages/tempDir/saved_text")
        except:
            print("file was not deleted")

        st.info('Done')
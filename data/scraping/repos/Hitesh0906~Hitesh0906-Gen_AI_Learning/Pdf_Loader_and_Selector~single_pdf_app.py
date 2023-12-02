
from dotenv import load_dotenv
import os
import shutil
import streamlit as st
import openai
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
import time
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

load_dotenv()
from PIL import Image

def main():
    openai.api_key = "xxxxxxxxxxx"
    os.environ['OPENAI_API_KEY'] = "xxxxxxxxxxx"
    img = Image.open(r"images.jpeg")
    st.set_page_config(page_title="Profile Selection.AI", page_icon=img)
    st.header("Know about your Candidate📄")
    pdf = st.file_uploader("Upload your PDF", type="pdf")  # , accept_multiple_files = True)
    print(pdf)
    query = st.text_input("Ask your Question about your PDF")
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chunks = text_splitter.split_text(text)

    # Split
        embedding_fun = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embedding_fun)

        if query:
            # docs = knowledge_base.similarity_search(query)
            docs = knowledge_base.max_marginal_relevance_search(query)
            llm = OpenAI(temperature=0)
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=query)
            st.success(response)

if __name__ == '__main__':
    main()



# Which profile is most suitable for a role of a Data Scientist and Why?

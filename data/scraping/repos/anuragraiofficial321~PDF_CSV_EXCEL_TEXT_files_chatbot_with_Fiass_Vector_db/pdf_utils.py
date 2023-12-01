import streamlit as st
from Secret_key import openai_api_key
import os 
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import getpass
from langchain.callbacks import get_openai_callback
import uuid

#os.environ['OPENAI_API_KEY'] = getpass.getpass(openaikey)


os.environ["OPENAI_API_KEY"]=openai_api_key   #add api key in form of sting "api_key" instead fo openai_api_key
llm=OpenAI()
    

def FiassVectordb(full_text,query):
    text_splitter = CharacterTextSplitter(        
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
             )
    chunks = text_splitter.split_text(full_text)
    st.success("PDF processing and indexing completed successfully!")
    #st.write(chunks)
    db_file = "fiass_doc_idx"
    #print("fiass before delete")
    #print("fiass after delete")
    #db = FAISS.from_documents(chunks, OpenAIEmbeddings())
    db = FAISS.from_texts(chunks, OpenAIEmbeddings())
    #print("save fiass")
    #db.save_local(db_file)
    
    #db = FAISS.load_local("fiass_doc_idx",OpenAIEmbeddings())
    docs=db.similarity_search(query)
    chain=load_qa_chain(llm,chain_type="stuff")
    response=chain.run(input_documents=docs, question=chain)
    return response


def pdftotext(pdf_file):
    pdf_reader =PdfReader(pdf_file)
    #txt_file = open("temp_pdf_text.txt", "w",encoding='utf-8')
    full_text=""
    for i in pdf_reader.pages:
        full_text+=i.extract_text()
    
    print("end pdf function")
    return full_text


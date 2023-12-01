import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from gdrive_helpers import delete_lecture_from_drive
from database import Database
from initialize import load_lecturenames
def create_faiss_store(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    text_splitter = CharacterTextSplitter(separator = "\n", chunk_size = 1000, 
                                            chunk_overlap  = 200, length_function = len)
    docs = text_splitter.split_documents(pages)
    for doc in docs:
        doc.metadata["page"] +=1
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    faiss_path = "tmp/uploaded_files/"
    if not os.path.isdir(faiss_path):
        os.makedirs(faiss_path)
    vectorstore.save_local(faiss_path)
    return vectorstore

def save_uploadedfile(uploadedfile,lecturename):
    upload_path = "tmp/uploaded_files"
    if not os.path.isdir(upload_path):
        os.makedirs(upload_path)
    with open(os.path.join(upload_path,f"{lecturename}.pdf"),"wb") as f:
        f.write(uploadedfile.getbuffer())
    return None

def delete_lecture(lecturename):
    delete_lecture_from_drive(lecturename)
    db = Database()
    db.execute_query("DELETE FROM filestorage WHERE username = %s AND lecture = %s",(st.session_state["username"],
                                                                                     lecturename))
    db.execute_query("DELETE FROM history WHERE username = %s AND lecture = %s",(st.session_state["username"],
                                                                                     lecturename))
    load_lecturenames()
    st.rerun()

def check_if_lecture_exists(lecturename):
    db = Database()
    uploaded_lectures = db.query("SELECT lecture from filestorage WHERE username = %s",
                                                    (st.session_state["username"],))
    uploaded_lectures = set([x[0] for x in uploaded_lectures])
    if lecturename in uploaded_lectures:
        return True
    else:
        return False
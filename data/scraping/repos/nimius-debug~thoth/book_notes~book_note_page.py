import database as db
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from qdrant_db import QdrantSingleton

##########################################################
def get_pdf_text(pdf):
    text = ""
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

@st.cache_data
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1100,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

##########################################################

def book_note_page(username):
    qdrant_singleton = QdrantSingleton()
    vector_store = qdrant_singleton.get_vector_store(username)
    st.title("Books/Notes")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Upload Your Documents to Thoth")
        st.write("Enhance Thoth's capabilities and personalize your academic assistance by uploading your study materials. \
            Each document you add contributes to your unique knowledge base, enabling Thoth AI to provide more targeted and \
            accurate academic support. Simply click the \"Upload Documents\" button to start empowering your learning experience.")
    with col2:
        
        try:
            uploaded_file = st.file_uploader("Upload Documents", type=["pdf"])
            if uploaded_file is not None:
                # Check if the file already exists
                if db.file_exists(st.session_state['username'], uploaded_file.name):
                    st.write("This file already exists. Skipping upload and semantic index creation.")
                else:
                    with st.spinner("Creating semantic index..."):
                        metadata = {'filename': uploaded_file.name}
                        
                        # Upload the file
                        result_message = db.put_file(st.session_state['username'], uploaded_file.name, uploaded_file)
    
                        # Get the text from the PDF
                        raw_text = get_pdf_text(uploaded_file)
                        text_chunks = get_text_chunks(raw_text)
                        
                        # Create metadata list with the same length as text_chunks put in vectorstore
                        metadatas_list = [metadata] * len(text_chunks)
                        vector_store = qdrant_singleton.get_vector_store(st.session_state['username'])
                        
                        try:
                            vector_store.add_texts(text_chunks, metadatas=metadatas_list)
                            st.toast("Text chunks added to vector store")
                        except Exception as e:
                            st.write(f"Error in add_texts: {e}")
                        
                        st.toast(result_message)
                
        except Exception as e:
            st.write(f"An error occurred: {e}")

    st.divider()
    st.subheader("Your Documents")

    try:
        with st.status("Loading your documents..."):
            try:
                db_files = db.list_files(st.session_state['username'])
            except Exception as e:
                st.write(f"Error in list_files: {e}")
        
        if not isinstance(db_files, list) or len(db_files) == 0:
            st.write("You have not uploaded any documents yet.")
        else:
            files_to_delete = []
            with st.form("file_deletion_form", clear_on_submit=True):
                for file in db_files:
                    checked = st.checkbox(f"{file}")
                    if checked:
                        files_to_delete.append(file)
                
                delete_button = st.form_submit_button("Delete Selected Files")
                if delete_button:
                    for file in files_to_delete:
                        print(f"Deleting {file}...")
                        with st.spinner(f"Deleting {file}..."):
                            result_message = db.delete_file(st.session_state['username'], file)
                            st.toast(result_message)
                            x = qdrant_singleton.delete_points_associated_with_file(filename=file, collection_name= username)
                            print(x)
                    st.toast(f"Deleted vectors associated with {file}")
                    st.rerun()
    except Exception as e:
        st.write(f"An error occurred: {e}")
   
    
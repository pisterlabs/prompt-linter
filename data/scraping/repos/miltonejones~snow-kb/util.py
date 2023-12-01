
import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader 
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS 

def convert_file_name(file_name): 
  file_name = file_name.split(".")[0] 
  words = file_name.split("_") 
  words = [word.capitalize() for word in words] 
  return " ".join(words)
  

def get_document_list():
  folder_path = './src'
  file_object = {}

  # Iterate over each file in the folder
  for file_name in os.listdir(folder_path):
    if os.path.isfile(os.path.join(folder_path, file_name)):
      # Normalize the file name
      normalized_key = convert_file_name(file_name)
      
      # Store the file path in the dictionary
      file_object[normalized_key] = './src/' + file_name

  # Sort the file list alphabetically
  sorted_files = dict(sorted(file_object.items()))

  return sorted_files

@st.cache_data
def load_description(_qa, pdf):
  st.session_state.chat_history = []
  result = _qa({"question": 'Give a one paragraph summary of the document, then Describe at least 10 and up to 25 subjects covered with one sentence each. give priority to any classes, functions or objects described. use markup where possible', "chat_history": []})
  # st.balloons()
  return result


@st.cache_data
def load_source_document(pdf_file): 
  base_name = os.path.basename(pdf_file)
  folder_name = os.path.splitext(base_name)[0]
  embeddings = OpenAIEmbeddings()

  with st.spinner(text="Loading "+ folder_name +"..."): 
    folder_path = f'db/{folder_name}'
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
      # Folder already exists, do something if needed
      return FAISS.load_local(folder_path, embeddings) 
    else: 
      loader = PyPDFLoader(pdf_file)
      chunks = loader.load_and_split()  
      db = FAISS.from_documents(chunks, embeddings)  
      db.save_local(folder_path)
      return db
    
def save_uploaded_file(uploaded_file):
  # Create the 'src' folder if it doesn't exist
  if not os.path.exists('src'):
    os.makedirs('src')
  
  # Save the uploaded file to the 'src' folder
  file_path = os.path.join('src', uploaded_file.name)
  with open(file_path, 'wb') as f:
    f.write(uploaded_file.getbuffer())
  
  load_source_document(file_path)
  return file_path

def initialize_session():
  session_keys = {
    "chat_history": [],
    "selected_option": '',
    "selected_desc": None,
    "llm_temparature": 0.7,
    "pdf_file": './src/client_api_reference.pdf' 
  }
 
  for key, value in session_keys.items():
    if key not in st.session_state:
      st.session_state[key] = value

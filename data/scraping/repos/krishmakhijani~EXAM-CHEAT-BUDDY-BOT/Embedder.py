
from langchain.embeddings import (
    LlamaCppEmbeddings, 
    HuggingFaceEmbeddings, 
    SentenceTransformerEmbeddings
)

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.vectorstores import FAISS
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import (
    PyPDFLoader,
    DataFrameLoader,
    GitLoader
  )
import pandas as pd
import nbformat
from nbconvert import PythonExporter
import os

def get_text_splits(text_file):
  """Function takes in the text data and returns the  
  splits so for further processing can be done."""
  with open(text_file,'r') as txt:
    data = txt.read()

  textSplit = RecursiveCharacterTextSplitter(chunk_size=500,
                                             chunk_overlap=100,
                                             length_function=len)
  doc_list = textSplit.split_text(data)
  return doc_list




def get_pdf_splits(pdf_file):
  """Function takes in the pdf data and returns the  
  splits so for further processing can be done."""
  
  loader = PyPDFLoader(pdf_file)
  pages = loader.load_and_split()  

  textSplit = RecursiveCharacterTextSplitter(chunk_size=850,
                                             chunk_overlap=200,
                                             length_function=len)
  doc_list = []
  #Pages will be list of pages, so need to modify the loop
  for pg in pages:
    pg_splits = textSplit.split_text(pg.page_content)# here we are giveing each page content with is a text
    doc_list.extend(pg_splits)

  return doc_list





def get_excel_splits(excel_file,target_col,sheet_name):
  trialDF = pd.read_excel(io=excel_file,
                          engine='openpyxl',
                          sheet_name=sheet_name)
  
  df_loader = DataFrameLoader(trialDF,
                              page_content_column=target_col)
  
  excel_docs = df_loader.load()

  return excel_docs




def embed_index(file_path , embed_fn, index_store):
  """Function takes in existing vector_store, 
  new doc_list and embedding function that is 
  initialized on appropriate model. Local or online. 
  New embedding is merged with the existing index. If no 
  index given a new one is created"""
  #check whether the doc_list is documents, or text

  doc_list = get_pdf_splits(file_path)

  try:
    faiss_db = FAISS.from_documents(doc_list, 
                              embed_fn)  
  except Exception as e:
    faiss_db = FAISS.from_texts(doc_list, 
                              embed_fn)
  
  if os.path.exists(index_store):
    local_db = FAISS.load_local(index_store,embed_fn)
    #merging the new embedding with the existing index store
    local_db.merge_from(faiss_db)
    print("Merge completed")
    local_db.save_local(index_store)
    print("Updated index saved")
  else:
    faiss_db.save_local(folder_path=index_store)
    print("New store created...")








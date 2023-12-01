#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os, sys, openai
from dotenv import load_dotenv
from glob import glob
import pypdfium2 as pdfium
from pyepsilla import cloud

# from langchain.document_loaders import TextLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter




# Extract text from pdf
def pdf2text(pdf_path):
  text = ''
  pdf = pdfium.PdfDocument(pdf_path)
  version = pdf.get_version()
  n_pages = len(pdf)
  for i in range(n_pages):
    page = pdf[i]
    textpage = page.get_textpage()
    text += textpage.get_text_range()

  return text


# Convert text to Embedding using openai model
# https://openai.com/blog/function-calling-and-other-api-updates
def get_embedding(text, model="text-embedding-ada-002"):
  text = text.replace("\n", " ")
  return openai.Embedding.create(input=text, model=model)['data'][0]['embedding']


# Convert text to Embedding
def text2embedding(text=None, pdf_name=None):
  documents = [Document(page_content=text, metadata={"source": pdf_name})]

  text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100, separators=["\n", "\n\n"])
  texts = [i.page_content for i in text_splitter.split_documents(documents)]
  
  embeddings=[]
  contexts=[]
  length = len(texts)

  for i in range(length):
    # text_str = texts[i].page_content
    embedding = get_embedding(texts[i], model="text-embedding-ada-002")
    embeddings.append(embedding)
    contexts.append( ''.join([texts[k] for k in range(max(0, i-1), min(i+2, length))]) )

  return embeddings, texts, contexts


# Store Embedding to Epsilla Cloud
def store2vectordb(records, table_name, db_id, project_id, api_key):
  # Connect to Epsilla Cloud
  client = cloud.Client(project_id, api_key)
  db = client.vectordb(db_id)
  status_code, response = db.insert( table_name=table_name, records=records)
  print(response)
  return status_code, response










if __name__ == "__main__":
  # Get Config of Vectordb on Epsilla Cloud
  load_dotenv() 

  project_id=os.getenv("PROJECT_ID")
  db_id=os.getenv("DB_ID")
  table_name=os.getenv("TABLE_NAME")
  api_key=os.getenv("EPSILLA_API_KEY")
  openai.api_key = os.getenv("OPENAI_KEY")



  record_num_total = 0

  # Get list of all pdf files in "./documents/"
  files = glob("./documents/*.pdf")
  
  for pdf in files[:1]:
    pdf_name = os.path.basename(pdf)

    # Extrace text from pdf
    text = pdf2text(pdf)

    # Convert text to embedding
    embeddings, texts, contexts = text2embedding(text, pdf_name)

    # Prepare record
    records = [ {"id": record_num_total+i+1, "doctitle": pdf_name, "embedding": embeddings[i], "text": texts[i],  "context": contexts[i]} for i in range(len(embeddings))]

    # Store records to epsilla cloud vectordb
    store2vectordb(records, table_name, db_id, project_id, api_key)

    # Accumulate the record num
    record_num_total += len(embeddings)
    print("record_num_total:", record_num_total)


from langchain.llms import OpenAI
from langchain import PromptTemplate, LLMChain
import random
import time
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
from langchain.document_loaders import TextLoader, PDFMinerLoader, CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import HuggingFaceInstructEmbeddings
from tqdm import tqdm
from langchain.vectorstores import FAISS
from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
import os
import glob
import argparse
import openai
from time import sleep
chunk_size = 512 #512
chunk_overlap = 50
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
global qa

from dotenv import load_dotenv
load_dotenv()

SECRET_IN_ENV = True

SECRET_TOKEN = os.environ["SECRET_TOKEN"] 
openai.api_key = SECRET_TOKEN
messages = []


"""# **Embeddings model object to vectorize documents **"""
def create_db():
    documents = []
    SECRET_IN_ENV = True
    SECRET_TOKEN = os.environ["SECRET_TOKEN"] 
    openai.api_key = SECRET_TOKEN
    messages = []    
    hf= OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=openai.api_key )

    a=glob.glob("source_documents/*.txt")
    for i in range(len(a)):
            print(a[i])
            documents.extend(TextLoader(a[i]).load())
            print(TextLoader(a[i]).load())
    
    a=glob.glob("source_documents/*.html")
    
    try:
      for i in range(len(a)):
            print(i)
    
            documents.extend(UnstructuredHTMLLoader(a[i]).load())
    except:
      print(i)
    
    a=glob.glob("source_documents/*.pdf")
    for i in range(len(a)):
    
            documents.extend(PDFMinerLoader(a[i]).load())
    
    a=glob.glob("source_documents/*.csv")
    for i in range(len(a)):
    
            documents.extend(CSVLoader(a[i]).load())
    
    a=glob.glob("source_documents/*.ppt")
    for i in range(len(a)):
    
            documents.extend(UnstructuredPowerPointLoader(a[i]).load())
    
    a=glob.glob("source_documents/*.pptx")
    for i in range(len(a)):
    
            documents.extend(UnstructuredPowerPointLoader(a[i]).load())
    
    
    a=glob.glob("source_documents/*.docx")
    for i in range(len(a)):
    
            documents.extend(UnstructuredWordDocumentLoader(a[i]).load())
    
    a=glob.glob("source_documents/*.ppt")
    for i in range(len(a)):
    
            documents.extend(UnstructuredPowerPointLoader(a[i]).load())
    
    
    chunk_size = 1024
    chunk_size = 512
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts_isb = text_splitter.split_documents(documents)
    
    db = FAISS.from_documents(texts_isb, hf)
    db.save_local("faiss_index_anupam")
    return db


def chat_gpt(question):
    db= create_db()
    SECRET_IN_ENV = True
    SECRET_TOKEN = os.environ["SECRET_TOKEN"] 
    openai.api_key = SECRET_TOKEN

    retriever = db.as_retriever(search_type='similarity', search_kwargs={"k": 5} )#do not increase k beyond 3, else
    llm = OpenAI(model='text-davinci-003',temperature=0, openai_api_key=openai.api_key)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
  
    
    query = question
    res = qa(query)
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful and friendly chatbot who answers based on contect provided in very friendly tone."},
        {"role": "user", "content": f"{res}"}
    ])
    answer= response["choices"][0]["message"]["content"]
    return answer





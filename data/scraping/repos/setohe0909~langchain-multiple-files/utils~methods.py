import streamlit as st

from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import  CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
# Vector database such as: faiss, pinecone
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import  ConversationBufferMemory
from langchain.chains  import ConversationalRetrievalChain
from utils.htmlTemplates import   bot_template, user_template
from langchain.llms import huggingface_hub

def get_pdf_text(pdf_docs):
  text = ""
  
  for pdf in pdf_docs:
    pdfReader = PdfReader(pdf)
    for page in pdfReader.pages:
      text += page.extract_text()  # Extract content from the page
  
  return text

def get_text_chunks(text):
  text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
  )
  chunk = text_splitter.split_text(text)
  
  return chunk

def get_vectorstore(text_chunks):
  # embeddings = OpenAIEmbeddings() #--> open-ai
  embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl") #--> https://huggingface.co/hkunlp/instructor-xl
  vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
  
  return vectorstore

def get_conversation_chain(vectorstore):
  llm= ChatOpenAI()
  
  # Using llm from huggingface
  #llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

  memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
  conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory
  )
  
  return conversation_chain

def handle_userinput(user_question):
  response = st.session_state.conversation({'question': user_question})
  
  st.session_state.chat_history = response['chat_history']
  
  for i, message in enumerate(st.session_state.chat_history):
    if i & 2 == 0:
      st.write(user_template.replace(
        "{{MSG}}", message.content), unsafe_allow_html=True)
    else:
      st.write(bot_template.replace(
        "{{MSG}}", message.content), unsafe_allow_html=True)

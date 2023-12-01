import streamlit as st
import datetime
import os
from os import environ
import PyPDF2

from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

from langchain.chat_models import ChatCohere
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.indexes.vectorstore import VectorstoreIndexCreator
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import CohereEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

cohere_api_key = st.secrets["COHERE_API_KEY"]
openai_api_key = st.secrets["OPENAI_API_KEY"]

def read_pdf_text(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""

    for page in pdf_reader.pages:
        text += page.extract_text()

    return text

def generate_response(doc_texts, cohere_api_key, query_text):

       # Define the character's name and description
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n"],chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_text(doc_texts)
    #print(texts)
    chat_model = ChatCohere(cohere_api_key=cohere_api_key, model='command',temperature=0.1)
    embeddings = CohereEmbeddings(cohere_api_key=cohere_api_key)
    # Create a vectorstore from documents
    vectorstore = Chroma.from_texts(texts, embeddings)
    # Create retriever interface
    
    # Initialize the chat model for the character
    chat_model = ChatCohere(cohere_api_key=cohere_api_key, model='command',temperature=0.0)
    user_input2 = "Why was Ollivander curious?"
    
    # Create the memory object
    memory=ConversationBufferMemory(
            memory_key='chat_history', return_messages=True)
    search = vectorstore.similarity_search(user_input2)
    print(search[0])
    
    # Create the QA chain with memory
    conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=chat_model,
            retriever= vectorstore.as_retriever(),
            memory=memory
        )
    response = conversation_chain({'question': user_input2})
    
    # Ask a question and get the answer
    print(response['chat_history'][1].content)
    
# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "You are a AI assistant created to help hiring managers review resumes and shortlist candidates. You have been provided with resumes and job descriptions to review. When asked questions, use the provided documents to provide helpful and relevant information to assist the hiring manager. Be concise, polite and professional. Do not provide any additional commentary or opinions beyond answering the questions directly based on the provided documents."}]

# Page title
st.set_page_config(page_title='Gforce Resume Assistant', layout='wide')
st.title('Gforce Resume Assistant')

# File upload
uploaded_files = st.file_uploader('Please upload you resume(s)', type=['txt'], accept_multiple_files=True)

# Query text
query_text = st.text_input('Enter your question:', placeholder='Select candidates based on experience and skills')

# Initialize chat placeholder as an empty list
if "chat_placeholder" not in st.session_state.keys():
    st.session_state.chat_placeholder = []

# Form input and query
if st.button('Submit', key='submit_button'):
    if openai_api_key.startswith('sk-'):
        if uploaded_files and query_text:
            documents = [uploaded_file.read().decode()]
            with st.spinner('Chatbot is typing...'):
                response = generate_response(documents, cohere_api_key, query_text)
                st.session_state.chat_placeholder.append({"role": "user", "content": query_text})
                st.session_state.chat_placeholder.append({"role": "assistant", "content": response})

            # Update chat display
            for message in st.session_state.chat_placeholder:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
        else:
            st.warning("Please upload one or more PDF files and enter a question to start the conversation.")

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    st.session_state.chat_placeholder = []
    uploaded_files.clear()
    query_text = ""
    st.empty()  # Clear the chat display

st.button('Clear Chat History', on_click=clear_chat_history)


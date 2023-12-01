# DocChatbot for multi docs
import os 
#import chromadb
from PyPDF2 import PdfReader
import pypdf
import tiktoken
import streamlit as st
from streamlit_chat import message
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chat_models import ChatOpenAI
import openai
from dotenv import load_dotenv
import tempfile

load_dotenv()
#openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = os.environ.get("OPEN_API_KEY")
os.environ["OPENAI_API_KEY"] = openai.api_key

model_id = "gpt-3.5-turbo"

llm=ChatOpenAI(model_name = model_id, temperature=0.1)

st.title('✨ AI Smart Query | AI智慧畅聊 ')
prompt = st.text_input("Welcome to AI Smart Query, questions will be answered based on the sources in our docs directory.\n\n欢迎来到AI智慧畅聊，基于上传的pdf文件资源进行高效QA问答。")


with st.sidebar:
    st.subheader("Upload your Documents Here: ")
    pdf_files = st.file_uploader("Choose your PDF Files and Press OK", type=['pdf'], accept_multiple_files=True)

loaders = []
for pdf_file in pdf_files:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(pdf_file.read())
        temp_file_path = temp_file.name
    loaders.append(PyPDFLoader(temp_file_path))

try:
    index = VectorstoreIndexCreator().from_loaders(loaders)
    index.save("./docchatbotindex")
    os.remove(temp_file_path)
except Exception as e:
# Handle the error, e.g., print an error message or return a default text
#    st.write("Documents not uploaded. Please upload your docs first and then enter your question.")
    os.environ["OPENAI_API_KEY"] = openai.api_key

#loaders = PyPDFLoader(pdf_files)
#index = VectorstoreIndexCreator().from_loaders([loaders])

if prompt:
    # stuff chain type sends all the relevant text chunks from the document to LLM    
    try:
        response = index.query(llm=llm, question = prompt, chain_type = 'stuff')   
    # Write the results from the LLM to the UI
        st.write("<br><i>" + response + "</i><hr>", unsafe_allow_html=True )
    #st.write("<b>" + prompt + "</b><br><i>" + response + "</i><hr>", unsafe_allow_html=True )
    except Exception as e:
# Handle the error, e.g., print an error message or return a default text
        st.write("Documents not uploaded or Unknown error.")

import os
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import AzureOpenAI
from langchain.embeddings import OpenAIEmbeddings
import re
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader
import tempfile
from langchain.vectorstores import FAISS
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from streamlit_chat import message
import streamlit as st
import openai

openai.api_type = 'azure'
openai.api_version = '2022-12-01'


OPEN_AI_API_KEY = '225ecf1a48ed49cab726774bdc6b675a'
OPENAI_API_BASE= 'https://ozguler.openai.azure.com/'
openai.api_key = OPEN_AI_API_KEY
openai.api_base = OPENAI_API_BASE   

# streamlit


# read the text
# Update with pypdf loader from langchain?
doc_reader = PdfReader('./Seneca-OntheShortnessofLife.pdf')

# read data from the file and put them into a variable called raw_text
raw_text = ''
for i, page in enumerate(doc_reader.pages):
    text = page.extract_text()
    if text:
        raw_text += text

print(len(raw_text))


# Splitting up the text into smaller chunks for indexing

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,  # striding over the text
    length_function=len,
)
texts = text_splitter.split_text(raw_text)

# s is input text


def normalize_text(s, sep_token=" \n "):
    s = re.sub(r'\s+',  ' ', s).strip()
    s = re.sub(r". ,", "", s)
    # remove all instances of multiple spaces
    s = s.replace("..", ".")
    s = s.replace(". .", ".")
    s = s.replace("\n", "")
    s = s.strip()
    return s


texts = list(map(normalize_text, texts))

os.environ["OPENAI_API_KEY"] = '225ecf1a48ed49cab726774bdc6b675a'
os.environ["OPENAI_API_BASE"] = 'https://ozguler.openai.azure.com/' 
openai_api_key = os.environ.get('OPENAI_API_KEY')
openai_api_base = os.environ.get('OPENAI_API_BASE')


embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", chunk_size=1)

vectorstore = Chroma.from_texts(texts, embeddings)

from langchain.chat_models import AzureChatOpenAI

    
chain = ConversationalRetrievalChain.from_llm(
    llm=AzureChatOpenAI(deployment_name='gpt35-turbo-ozguler',openai_api_version="2023-03-15-preview"),
    retriever=vectorstore.as_retriever())



def conversational_chat(query):

    result = chain({"question": query,
                    "chat_history": st.session_state['history']})
    

    st.session_state['history'].append((query, result["answer"]))

    return result["answer"]

if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello ! Ask me anything about " + " 🤗"]

if 'past' not in st.session_state:
    st.session_state['past'] = ["Hey ! 👋"]
    
#container for the chat history
response_container = st.container()
#container for the user's text input
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        
        user_input = st.text_input("Query:", placeholder="Talk about your csv data here (:", key='input')
        submit_button = st.form_submit_button(label='Send')
        
    if submit_button and user_input:
        output = conversational_chat(user_input)
        
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)
        

if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
    

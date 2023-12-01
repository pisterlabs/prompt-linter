import streamlit as st
from pdf_qa import PdfQA
from pathlib import Path
from tempfile import NamedTemporaryFile
import time
import shutil
from langchain.vectorstores import Chroma
from constants import *
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PDFPlumberLoader
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter
from langchain.document_loaders import PyPDFLoader
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain import HuggingFacePipeline
from langchain.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from constants import *
from transformers import AutoTokenizer
import torch
import re
import PIL.Image
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
import tempfile
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template






# Streamlit app code
st.set_page_config(
    page_title='Q&A Bot for PDFs',
    page_icon='🔖',
    layout='wide',
    initial_sidebar_state='auto',
)
st.write(css, unsafe_allow_html=True)




from streamlit_extras.colored_header import colored_header

colored_header(
    label="Welcome to  Q&A Bot for CSV",
    description="Upload a CSV and ask questions about it, or if you have already uploaded a CSV, ask questions about it.",
    color_name="violet-70",
)



if "pdf_qa_model" not in st.session_state:
    st.session_state["pdf_qa_model"]:PdfQA = PdfQA() ## Intialisation

## To cache resource across multiple session 
@st.cache_resource
def load_llm(llm,load_in_8bit):

    if llm == LLM_OPENAI_GPT35:
        pass
    elif llm == LLM_FLAN_T5_SMALL:
        return PdfQA.create_flan_t5_small(load_in_8bit)
    elif llm == LLM_FLAN_T5_BASE:
        return PdfQA.create_flan_t5_base(load_in_8bit)
    elif llm == LLM_FLAN_T5_LARGE:
        return PdfQA.create_flan_t5_large(load_in_8bit)
    elif llm == LLM_FASTCHAT_T5_XL:
        return PdfQA.create_fastchat_t5_xl(load_in_8bit)
    elif llm == LLM_FALCON_SMALL:
        return PdfQA.create_falcon_instruct_small(load_in_8bit)
    else:
        raise ValueError("Invalid LLM setting")

## To cache resource across multiple session
@st.cache_resource
def load_emb(emb):
    if emb == EMB_INSTRUCTOR_XL:
        return PdfQA.create_instructor_xl()
    elif emb == EMB_OPENAI_ADA:
        return OpenAIEmbeddings()
    elif emb == EMB_SBERT_MPNET_BASE:
        return PdfQA.create_sbert_mpnet()
    elif emb == EMB_SBERT_MINILM:
        pass ##ChromaDB takes care
    else:
        raise ValueError("Invalid embedding setting")




def get_csv_text(csv_path):
 documents=[]
 for csv in csv_path:
   
    with NamedTemporaryFile(delete=False) as tmp:
            shutil.copyfileobj(csv, tmp)
            tmp_path = Path(tmp.name)

    loader =CSVLoader(file_path=str(tmp_path), encoding="utf-8", csv_args={
                'delimiter': ','})
    documents.extend(loader.load())
    return documents
 print(documents)
 
   
def get_text_chunks(documents):
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    # text_splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=10, encoding_name="cl100k_base")  # This the encoding for text-embedding-ada-002
    text_splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=10)  # This the encoding for text-embedding-ada-002
    texts = text_splitter.split_documents(texts)
    return texts   
def vector_db_pdf(csv_path) -> None:
    """
    creates vector db for the embeddings and persists them or loads a vector db from the persist directory
    """
    #persist_directory = PdfQA.config.get("persist_directory",None)
    print("#################################")
    
    #if PDF is not present then load from persist directory else condition otherwise use pdf to generate persist vector DB
    #if len(pdf_path)>0:
    if csv_path is not None:
    #print(persist_directory)
        documents=get_csv_text(csv_path)
        texts =get_text_chunks(documents)            ## 3. Create Embeddings and add to chroma store
        ##TODO: Validate if PdfQA.embedding is not None
        vector_db=Chroma.from_documents(documents=texts, embedding=st.session_state["pdf_qa_model"].embedding, persist_directory="storage")
        vector_db.persist()    
        
    else:
        #Use from persist
        vector_db=Chroma(persist_directory="storage", embedding_function=st.session_state["pdf_qa_model"].embedding)
    return vector_db
    #else:
    #    raise ValueError("NO PDF found")    
    


st.title(" Self hosted LLMs")


def handle_userinput(user_question):
    response = st.session_state["pdf_qa_model"].conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

with st.sidebar:
    
    csv_path= st.file_uploader("**Upload CSV**", type="csv" , accept_multiple_files=True)
    emb = st.radio("**Select Embedding Model**", [EMB_INSTRUCTOR_XL, EMB_SBERT_MPNET_BASE, EMB_SBERT_MINILM, EMB_OPENAI_ADA], index=1)
    llm = st.radio("**Select LLM Model**", [LLM_FASTCHAT_T5_XL, LLM_FLAN_T5_SMALL, LLM_FLAN_T5_BASE, LLM_FLAN_T5_LARGE, LLM_FLAN_T5_XL, LLM_FALCON_SMALL, LLM_OPENAI_GPT35], index=2)
    load_in_8bit = st.radio("**Load 8 bit**", [True, False], index=1)

    if st.button("Set Params"):
        with st.spinner(text="Setting Params.."):
            st.session_state["pdf_qa_model"].config = {
                "embedding": emb,
                "llm": llm,
                "load_in_8bit": load_in_8bit
            }
            if "conversation" not in st.session_state["pdf_qa_model"].conversation:
                st.session_state["pdf_qa_model"].conversation = None
            st.session_state["pdf_qa_model"].embedding = load_emb(emb)
            st.session_state["pdf_qa_model"].llm = load_llm(llm, load_in_8bit)
            st.session_state["pdf_qa_model"].init_embeddings()
            st.session_state["pdf_qa_model"].init_models()
            st.session_state["pdf_qa_model"].vectordb = vector_db_pdf(csv_path)
            st.session_state["pdf_qa_model"].conversation = st.session_state["pdf_qa_model"].get_conversation_chain()
            #st.session_state.chat_history.clear()  # Clear chat history when setting new parameters
            st.sidebar.success("Parameter generated successfully")

    if st.button("Upload and Set Params") and csv_path is not None:
        with st.spinner(text="Uploading csv and Generating Embeddings.."):
            st.session_state["pdf_qa_model"].config = {
                "embedding": emb,
                "llm": llm,
                "load_in_8bit": load_in_8bit
            }
            st.session_state["pdf_qa_model"].embedding = load_emb(emb)
            st.session_state["pdf_qa_model"].llm = load_llm(llm, load_in_8bit)
            st.session_state["pdf_qa_model"].init_embeddings()
            st.session_state["pdf_qa_model"].init_models()
            st.session_state["pdf_qa_model"].vectordb = vector_db_pdf(csv_path)
            st.session_state["pdf_qa_model"].conversation = st.session_state["pdf_qa_model"].get_conversation_chain()
            #st.session_state.chat_history.clear()  # Clear chat history when uploading new PDF
            st.sidebar.success("CSV uploaded and parameter generated successfully")


question = st.text_input('Ask a question', 'What is this document?')

if st.button("Answer"):
    try:
        handle_userinput(question)

    
    
    except Exception as e:
        st.error(f"Not able to answer the question right now: {str(e)}")

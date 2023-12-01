import os, json, openai, pickle
from PyPDF2 import PdfReader
from langchain.document_loaders import PyPDFLoader 
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain import HuggingFaceHub
from langchain.chains import RetrievalQA
# import streamlit as st
# from htmlTemplates import bot_template, user_template, css

# CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')
# with open(CONFIG_PATH, 'r') as f:
#     CONFIG = json.load(f)

def get_pdf_text(input_path):
    """
    Extract text from a given PDF file or from all PDF files within a specified directory.

    Parameters:
    - input_path (str): Path to the PDF file or directory containing PDF files.

    Returns:
    - str: Extracted text from the PDF file(s).

    Raises:
    - ValueError: If the provided path is neither a PDF file nor a directory containing PDF files.
    - FileNotFoundError: If the provided path does not exist.
    """

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"The specified path '{input_path}' does not exist.")

    # Extract text from a single PDF
    def extract_from_pdf(pdf_path):
        try:
            reader = PdfReader(pdf_path)
            return ''.join(page.extract_text() for page in reader.pages)
        except Exception as e:
            print(f"Error reading '{pdf_path}': {e}")
            return ""

    # If the input is a single PDF file
    if os.path.isfile(input_path) and input_path.endswith('.pdf'):
        print(f"Reading '{input_path}'...")
        print("Finished: 1 PDF file loaded")
        return extract_from_pdf(input_path)

    # If the input is a directory
    elif os.path.isdir(input_path):
        pdf_files = [os.path.join(input_path, file) for file in os.listdir(
            input_path) if file.endswith('.pdf')]

        if not pdf_files:
            raise ValueError("No PDF files found in the specified directory.")

        print(f"Reading {len(pdf_files)} PDF files:")
        for file in pdf_files:
            print(f"Reading file: {file} ...")
        print(f"Finished: {len(pdf_files)} PDF file(s) loaded")
        return ''.join(extract_from_pdf(pdf_file) for pdf_file in pdf_files)

    else:
        raise ValueError(
            "The provided path is neither a PDF file nor a directory containing PDF files.")

def get_chunk_text(text):
    
    text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap = 200,
    length_function = len
    )

    chunks = text_splitter.split_text(text)
    print(f"Finished: {len(chunks)} chunks generated from PDF texts.") 
    return chunks

def get_vector_store(text_chunks, embedding="openai"):
    """
    get vector store from text chunks, with a specified embedding.

    emebedding: str 
        "openai" or hugging face embedding id, for example "hkunlp/instructor-xl"

    """
    
    # For OpenAI Embeddings
    if embedding == "openai":
        embeddings = OpenAIEmbeddings()
    else:
        # For Huggingface Embeddings
        embeddings = HuggingFaceInstructEmbeddings(model_name = embedding)

    vectorstore = FAISS.from_texts(texts = text_chunks, embedding = embeddings)
    
    return vectorstore

def get_conversation_chain(vector_store, model='openai'):
    
    if (model == 'openai'):
        # OpenAI Model
        llm = ChatOpenAI(temperature=0.0, model_name='gpt-3.5-turbo', max_tokens=256)
    else:
        # HuggingFace Model
        llm = HuggingFaceHub(repo_id=model, model_kwargs={"temperature":0.000001, "max_length":256})

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    openmp_qa_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vector_store.as_retriever(),
        memory = memory
    )

    return openmp_qa_chain


def get_retrievalQA(vector_store, model='openai'):
    
    if (model == 'openai'):
        # OpenAI Model
        llm = ChatOpenAI(temperature=0.0, model_name='gpt-3.5-turbo', max_tokens=256)
    else:
        # HuggingFace Model
        llm = HuggingFaceHub(repo_id=model, model_kwargs={"temperature":0.000001, "max_length":256})

    openmp_qa_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type="stuff",
        retriever = vector_store.as_retriever(),
    )

    return openmp_qa_chain


def store_embeddings(docs, embeddings, store_name, path):

    vectorStore = FAISS.from_documents(docs, embeddings)

    with open(f"{path}/faiss_{store_name}.pkl", "wb") as f:
        pickle.dump(vectorStore, f)
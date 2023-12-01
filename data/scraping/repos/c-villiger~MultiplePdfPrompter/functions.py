"""
Imports
"""
import textwrap
# Document loaders
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredMarkdownLoader
# Splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
# Embeddings and models
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS, Chroma
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryMemory
# Chains
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains import RetrievalQA
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import RetrievalQAWithSourcesChain
import langchain
# Utils
import os
from termcolor import colored

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub

"""
Text processing functions
"""


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


"""
Converstation chain functions
"""


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def conversation_chain(vectorstore, chain_type, k, own_knowledge=False, show_pages=False):

    # Import API Key
    from apikey import API_KEY
    os.environ["OPENAI_API_KEY"] = API_KEY

    """
    Prompts
    """

    # Define Chain
    if own_knowledge:
        prompt_template = """Use the following pieces of chat history and context to answer the question at the end. \
            If the answer does not become clear from the context, you can also use your own knowledge. \
            If you use your own knowledge, please indicate this clearly in your answer. \

        Context:
        {context}

        {question}
        Helpful answer:"""

    if not own_knowledge:

        prompt_template = """Use the following pieces of chat history and context to answer the question at the end. \
            Do NOT use your own knowledge and give the best possible answer from the context.\
        
        Context:
        {context}

        {question}
        Helpful answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain_type_kwargs = {"prompt": PROMPT}

    """
    Chains
    """
    # QA chain that is adaptable
    # Amount of returned documents k-i -> makes it adaptable. Otherwise, it would always return k documents and the output would be the same.
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": k})

    # Define retrieval chain
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0),
        chain_type=chain_type,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs
    )

    return qa


"""
Functions for graphics
"""


def print_boxed_header(header, color):
    header_length = len(header)
    box_top = "╒" + "═" * (header_length + 2) + "╕"
    box_middle = f"│ {header} │"
    box_bottom = "╘" + "═" * (header_length + 2) + "╛"

    print(colored(box_top, color))
    print(colored(box_middle, color))
    print(colored(box_bottom, color))

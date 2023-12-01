import streamlit as st
import langchain
from langchain.schema import SystemMessage, AIMessage, HumanMessage
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from config import *
import time
import config
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS, Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI

def get_pdf_text(pdf_docs):
    text = ""
    reader = PdfReader(pdf_docs)
    for page in reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator= "\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
        )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY))
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.Jeff_conversation({'question': user_question})
    return response

def send_message_to_openai(bot, message):
    st.session_state[bot + "_chat_history"].append(HumanMessage(content=message))
    chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
    try:
        with get_openai_callback() as cb:
            response = chat(st.session_state[bot + "_chat_history"])
            st.session_state[bot + "_chat_history"].append(AIMessage(content=response.content))
            st.session_state.addtoken = cb.total_tokens
            st.session_state.token += cb.total_tokens
            return response.content
    except Exception as error_message:
        return error_message



def reset(bot_name):
    st.session_state[bot_name + "_chat_history"] = []
    st.session_state[bot_name + "_chat_history"] += config.predefinedMessage[bot_name]


def Doris(command="", user_message=""):
    st.session_state.current_bot = "Doris"
    if "Doris_chat_history" not in st.session_state:
        reset("Doris")

    match command:
        case "reset":
            reset("Doris")
            return
        
        case "get history":
            return st.session_state.Doris_chat_history
    
        case "chat":
            response = send_message_to_openai("Doris", user_message)
            return response
    return KeyError("Command unfound")

def Emily(command="", user_message=""):
    st.session_state.current_bot = "Emily"
    st.session_state["Emily_chat_history"] = []
    st.session_state["Emily_chat_history"] += config.predefinedMessage["Emily"]

    match command:
        case "reset":
            reset("Emily")
            return
        
        case "get history":
            return st.session_state.Emily_chat_history
    
        case "chat":
            response = send_message_to_openai("Emily", user_message)
            return response
    return KeyError("Command unfound")

def Alex(command="", user_message=""):
    st.session_state.current_bot = "Alex"
    if "Emily_chat_history" not in st.session_state:
        reset("Alex")

    match command:
        case "reset":
            reset("Alex")
            return
        
        case "get history":
            return st.session_state.Emily_chat_history
    
        case "chat":
            response = send_message_to_openai("Alex", user_message)
            return response
    return KeyError("Command unfound")

def Jerry(command="", user_message=""):
    st.session_state.current_bot = "Jerry"
    if "Jerry_chat_history" not in st.session_state:
        reset("Jerry")

    match command:
        case "reset":
            reset("Jerry")
            return
        
        case "get history":
            return st.session_state.Jerry_chat_history
    
        case "chat":
            response = send_message_to_openai("Jerry", user_message)
            return response
    return KeyError("Command unfound")

def Jeff(command="", user_message="", pdf = None):
    st.session_state.current_bot = "Jeff"
    if "Jeff_chat_history" not in st.session_state:
        reset("Jeff")

    match command:
        case "reset":
            reset("Jeff")
            return
        
        case "get history":
            return st.session_state.Jeff_chat_history

        case "chat":
            #response = send_message_to_openai("Jeff", user_message)
            response = handle_userinput(user_message)
            st.session_state.Jeff_chat_history = response['chat_history']
            return response['answer']

        case "upload_pdf":
            raw_text = get_pdf_text(pdf)

            text_chunks = get_text_chunks(raw_text)

            vectorstore = get_vectorstore(text_chunks)
            
            st.session_state.Jeff_conversation = get_conversation_chain(vectorstore)
    
    return KeyError("Command unfound")




bot = {
    "Doris" : Doris,
    "Emily" : Emily,
    "Alex" : Alex,
    "Jerry" : Jerry,
    "Jeff" : Jeff,
}
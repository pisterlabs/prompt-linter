from dotenv import load_dotenv

import streamlit as st
from streamlit_chat import message

from PyPDF2 import PdfReader

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )
    return conversation_chain


def handle_user_input(user_question):
    response = st.session_state.conversation({"question": user_question})["answer"]
    st.session_state.requests.append(user_question)
    st.session_state.responses.append(response)


def main():
    load_dotenv()

    st.set_page_config(page_title="QA ChatBot", 
                       page_icon=":robot:",
                       initial_sidebar_state="auto",
                       layout="centered")
    
    st.markdown("<h1 style='text-align: center; color: green;'>KubatAI</h1>", unsafe_allow_html=True)
    st.subheader("Ваш Бот-помощник")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "responses" not in st.session_state:
        st.session_state["responses"] = []
    if "requests" not in st.session_state:
        st.session_state["requests"] = []
    if "instructions" not in st.session_state:
        st.session_state["instructions"] = ["Пожалуйста, загрузите и обработайте PDF-файл(ы)."]

    response_container = st.container()
    textcontainer = st.container()

    with textcontainer:
        user_question = st.text_input("Введите ваш вопрос: ", key="input")
        if user_question:
            with st.spinner(""):
                if st.session_state.conversation is not None:
                    handle_user_input(user_question)
                else:
                    st.warning("Пожалуйста, сначала загрузите и обработайте PDF-файл.")
    with response_container:
        message(st.session_state["instructions"][0])

        for i in range(len(st.session_state["responses"])):            
            message(st.session_state["responses"][i], key=str(i))
            if i < len(st.session_state["requests"]):
                message(st.session_state["requests"][i], is_user=True, key=str(i) + "_user")
            
        

    with st.sidebar:
        st.subheader("Ваши Документы")
        pdf_docs = st.file_uploader(
            "Выберите PDF файлы и нажмите 'Обработать'", accept_multiple_files=True
        )
        if st.button("Обработать"):
            with st.spinner(""):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.session_state.responses.append("Отлично! Можете задавать вопросы.")
                st.rerun()

if __name__ == "__main__":
    main()

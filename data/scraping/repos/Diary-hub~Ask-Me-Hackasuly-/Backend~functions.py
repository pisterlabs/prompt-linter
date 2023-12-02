from sqlalchemy import null
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import googletrans
import json
import os

from google.cloud import translate_v2

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"Backend\googlekey.json"


def transtlate(msg, sc="en"):
    transtlate_client = translate_v2.Client()

    text = msg

    responce = transtlate_client.translate(text, target_language=sc)
    # print(responce["translatedText"])
    return responce["translatedText"]


transtlate("whatsup")


def getChunksOfText(raw_texts):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(raw_texts)
    return chunks


def getAllTexts(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def getVectorStore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorStore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorStore


def getConversationChain(vectorStore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversationChain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorStore.as_retriever(), memory=memory
    )
    return conversationChain


def getAnswer(user_question, conversation):
    responce = conversation(
        {
            "question": "(answer based on the contexct provided which is called PDF, dont answer questions not related to the pdf) "
            + user_question
        }
    )
    chat_history = responce["chat_history"]

    for i, msg in enumerate(chat_history):
        if i % 2 == 0:
            print(msg.content)
            st.write(
                # user_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True
            )
        else:
            print(msg.content)

            st.write(
                # bot_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True
            )
    return chat_history


def handleUserInput(user_question):
    responce = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = responce["chat_history"]

    for i, msg in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(
                # user_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True
            )
        else:
            st.write(
                # bot_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True
            )


def main():
    load_dotenv()
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.set_page_config(page_title="ASKME", page_icon="books:")
    # st.write(css, unsafe_allow_html=True)
    st.header("CHAT WITH YOUR... EVERYTHINHG")
    user_question = st.text_input("ASK ANYTHING...")

    if user_question:
        handleUserInput(user_question)

    with st.sidebar:
        st.subheader("Your Files:")
        pdf_docs = st.file_uploader(
            'Upload PDF(s) and Click "Process"', accept_multiple_files=True
        )

        if st.button("Process"):
            with st.spinner("Processing..."):
                # get PDFS
                raw_texts = getAllTexts(pdf_docs)

                # get chunks
                text_chunks = getChunksOfText(raw_texts)

                # create vectorStore => using embeddings
                vectorStore = getVectorStore(text_chunks)

                # create conversation chains
                st.session_state.conversation = getConversationChain(vectorStore)


if __name__ == "__main__":
    # main()
    null


# import requests

# load_dotenv()
# headers = {
#     "accept": "application/json",
#     "Authorization": os.getenv("ASOSOFT"),
#     # requests won't add a boundary if this header is set when you pass files=
#     # 'Content-Type': 'multipart/form-data',
# }

# files = {
#     "audio": open("Backend\Recording.mp3", "rb"),
# }

# response = requests.post(
#     "https://api.kurdishspeech.com/api/v1/asr/speech-to-text",
#     headers=headers,
#     files=files,
# )

# print(response.json())

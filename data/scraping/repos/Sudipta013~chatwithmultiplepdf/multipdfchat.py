import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
import os

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
    #embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    #llm = ChatOpenAI()
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512},huggingfacehub_api_token=st.secrets["HUGGINGFACEHUB_API_TOKEN"])

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
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
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    #CSS
    hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
    st.markdown(hide_st_style, unsafe_allow_html=True)

    bg = """
        <style> [data-testid="stAppViewContainer"]
        {
            background: rgb(6,36,39);
        }
        </style>
        """
    sb = """
        <style>[data-testid="stSidebar"]
        {
        background: rgb(42, 52, 65);
        }
        </style>
    """
    st.markdown(sb,unsafe_allow_html=True)
    st.markdown(bg, unsafe_allow_html=True)
    
    # Add the yellow bottom bar
    bottom_bar_html = """
    <style>
    .bottom-bar {
        background-color: #FFA500;
        padding: 5px;
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        text-align: center;
        font-family: 'Russo One';
        font-size: 20px;
    }
    </style>
    <div class="bottom-bar">
        <span style="color: white; font-weight: bold;">The Techie Indians</span>
    </div>
    """
    st.markdown(bottom_bar_html, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.markdown("<h1 style='text-align: center; font-family:Abril Fatface ; -webkit-text-stroke: 1px black ;font-size: 50px; padding-bottom: 15px; color: rgb(255, 255, 255) ;'>Chat with multiple PDFs</h1>", unsafe_allow_html=True)
    st.markdown("""<h5 style='text-align: center;font-size:18px;color: rgba(255,255,255,0.4); padding-top: 15px'>
                Chat with multiple PDF AI tool is an interactive application that allows users to upload and communicate with multiple PDF documents using large language models. 
                The tool facilitates natural language-based interactions and offers a user-friendly interface to extract and converse with the content of the uploaded PDFs in real-time.
                </h5>""",unsafe_allow_html=True)
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.markdown("<h1 style='text-align: Left; font-family:Abril Fatface ;font-size: 32px; padding-bottom: 1px; color: rgb(255,255,255) ;'>Your Documents:</h1>", unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: Left ;font-size: 18px; padding-bottom: 0px; color: rgb(255, 165, 0) ;'>Upload your file and click process</h1>", unsafe_allow_html=True)
        pdf_docs = st.file_uploader("", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)


if __name__ == '__main__':
    main()

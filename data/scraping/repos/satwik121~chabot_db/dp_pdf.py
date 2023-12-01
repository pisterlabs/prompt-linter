import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css ,bot_template, user_template
# from langchain.llms import HuggingFaceHub
# import tabula
from io import BytesIO
import streamlit as st
import openai
from streamlit_chat import message
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
# key = "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"
key = st.secrets['key']

if "conversation" not in st.session_state:
        st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = None
if "txt" not in st.session_state:
    st.session_state.txt = None
# if "tb" not in st.session_state:
#     st.session_state.tb = None
# if "agree" not in st.session_state:
#     st.session_state.agree = None
if "radio" not in st.session_state:
    st.session_state.radio = None
if "im" not in st.session_state:
    st.session_state.im = None
    
    
def extract_tables_from_pdf(pdf_docs):

    extracted_tables = []

    if pdf_docs is not None:
        # Ensure pdf_docs is a list
        if not isinstance(pdf_docs, list):
            pdf_docs = [pdf_docs]

        for pdf_doc in pdf_docs:
            # Use BytesIO to create a file-like object from the uploaded PDF data
            pdf_file = BytesIO(pdf_doc.read())

            # Use Tabula to extract tables
            try:
                tables = tabula.read_pdf(pdf_file, pages='all', multiple_tables=True)
                #st.write(tables)
                extracted_tables.extend(tables)
            except Exception as e:
                st.error(f"Error extracting tables from PDF: {str(e)}")

    return extracted_tables

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
    embeddings = OpenAIEmbeddings(openai_api_key=key)
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(openai_api_key=key)
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
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




def livegpt2(user_input):

    chat = ChatOpenAI(temperature=0,openai_api_key=key)

    # initialize message history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful assistant.")
        ]

    # handle user input
    if user_input:
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.spinner("Thinking..."):
            response = chat(st.session_state.messages)
        st.session_state.messages.append(
            AIMessage(content=response.content))

    # display message history
    messages = st.session_state.get('messages', [])
    for i, msg in enumerate(messages[1:]):
        if i % 2 == 0:
            message(msg.content, is_user=True, key=str(i) + '_user')
        else:
            message(msg.content, is_user=False, key=str(i) + '_ai')

def pdf():

    load_dotenv()
    # st.set_page_config(page_title="Docs Miner📚", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    with st.container():
        col1, col2 = st.columns([0.2, 0.7])
        with col1:
            st.image('sat.png', width=300)
        with col2:
            st.markdown("""
                <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 35vh;">
                    <h1> Welcome to Document Miner📚!</h1>
                    <!-- Additional content goes here -->
                </div>
            """, unsafe_allow_html=True)
    st.header("Ask Question's about your documents")
    if "user_question" not in st.session_state:
        st.session_state.user_question = None
    st.session_state.user_question = st.text_input("Ask a question about your documents:")
    # clear_button = st.button("Clear Input")
    # if clear_button:
    #     # Clear the input field by resetting the session state
    #     st.session_state.user_question = ""

    
    

    with st.sidebar:

        st.session_state.radio = st.radio("Select Action",["Private Mode","Public Mode"])

    if st.session_state.user_question:
        
        if st.session_state.conversation == None :
            st.write("Upload The Pdf First")

        else :
            if st.session_state.radio == "Private Mode" :
                handle_userinput(st.session_state.user_question)

            elif st.session_state.radio == "Public Mode" :
                st.warning("Chatbot in Live Mode",icon="⚠️")
                livegpt2(st.session_state.user_question)

            else :
                st.write("Choose Mode")

    with st.sidebar:

        st.subheader("Process The Docs Here!")
        pdf_docs = st.file_uploader(
            "", accept_multiple_files=True)
        # st.write("pdf_docs")
        # st.write(pdf_docs)
        if st.button("Process"):
            if len(pdf_docs) == 0:
                st.write("🤖Upload the Pdf First📑")
            else:
                with st.spinner("Processing"):
                    
                    #get tables
                    # st.session_state.tb = extract_tables_from_pdf(pdf_docs)

                    # get pdf text
                    raw_text = get_pdf_text(pdf_docs)
                    st.session_state.txt = raw_text
                    st.write(raw_text)

                    # get the text chunks
                    text_chunks = get_text_chunks(raw_text)

                    # create vector store
                    vectorstore = get_vectorstore(text_chunks)

                    # create conversation chain
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    
                # st.session_state.radio = st.radio("Select Action",["Get Tables","Get Data"])


if __name__ == '__main__':
    pdf()
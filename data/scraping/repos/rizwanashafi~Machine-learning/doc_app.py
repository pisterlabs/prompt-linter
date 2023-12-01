import streamlit as st
import os
from langchain.document_loaders.csv_loader import CSVLoader
import docx
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain import HuggingFaceHub
from streamlit_chat import message
from langchain.callbacks import get_openai_callback
from sentence_transformers import SentenceTransformer

openapi_key = st.secrets["OPENAI_API_KEY"]

# "with" notation
def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with your file")
    st.header("DocumentGPT")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        uploaded_files =  st.file_uploader("Upload your file",type=['csv'],accept_multiple_files=True)
        openai_api_key = openapi_key
        # openai_api_key = st.text_input("OpenAI API Key", key=openapi_key , type="password")
        process = st.button("Process")
    if process:
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
        files_text = get_files_text(uploaded_files)
        st.write("File loaded...")
        # get text chunks
        text_chunks = get_text_chunks(files_text)
        st.write("file chunks created...")
        # create vetore stores
        vetorestore = get_vectorstore(text_chunks)
        st.write("Vectore Store Created...")
         # create conversation chain
        st.session_state.conversation = get_conversation_chain(vetorestore,openai_api_key) #for openAI

        st.session_state.processComplete = True

    if  st.session_state.processComplete == True:
        user_question = st.chat_input("Ask Question about your files.")
        if user_question:
            handel_userinput(user_question)

# Function to get the input file and read the text from it.
def get_files_text(uploaded_files):
    text = ""
    for uploaded_file in uploaded_files:
        split_tup = os.path.splitext(uploaded_file.name)
        file_extension = split_tup[1]
        if file_extension == ".csv":
            text += get_csv_text(uploaded_file)
        elif file_extension == ".docx":
            text += get_docx_text(uploaded_file)
        else:
            text += get_csv_text(uploaded_file)
    return text

# Function to read PDF Files
def get_pdf_text(scv):
    pdf_reader = csvReader(pdf)
    text = ""
    for page in csv_reader.pages:
        text += page.extract_text()
    return text

def get_docx_text(file):
    doc = docx.Document(file)
    allText = []
    for docpara in doc.paragraphs:
        allText.append(docpara.text)
    text = ' '.join(allText)
    return text

def get_csv_text(file):
    return "a"

def get_text_chunks(text):
    # spilit ito chuncks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=900,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    # Using the hugging face embedding models
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # creating the Vectore Store using Facebook AI Semantic search
    knowledge_base = FAISS.from_texts(text_chunks,embeddings)
    return knowledge_base

def get_conversation_chain(vetorestore,openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name = 'gpt-3.5-turbo',temperature=0)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vetorestore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handel_userinput(user_question):
    with get_openai_callback() as cb:
        response = st.session_state.conversation({'question':user_question})
    st.session_state.chat_history = response['chat_history']

    # Layout of input/response containers
    response_container = st.container()

    with response_container:
        for i, messages in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                message(messages.content, is_user=True, key=str(i))
            else:
                message(messages.content, key=str(i))


if __name__ == '__main__':
    main()






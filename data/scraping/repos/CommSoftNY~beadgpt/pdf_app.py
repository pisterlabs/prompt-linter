# Import necessary modules
import pandas as pd
import streamlit as st 
from PIL import Image
from PyPDF2 import PdfReader
import openai

from langchain.embeddings import OpenAIEmbeddings, SentenceTransformerEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.memory import ConversationBufferWindowMemory
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

home_privacy = "We value and respect your privacy. To safeguard your personal details, we utilize the hashed value of your OpenAI API Key, ensuring utmost confidentiality and anonymity. Your API key facilitates AI-driven features during your session and is never retained post-visit. You can confidently fine-tune your research, assured that your information remains protected and private."


# Page configuration for Simple PDF App
st.set_page_config(
    page_title="BEAD Louisiana Q&A with AI",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
    )

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
#OPENAI_API_KEY = # st.sidebar.text_input("Enter Your OpenAI API Key:", type="password")
st.sidebar.subheader("Model Selection")
preferred_model='gpt-4'
llm_model_options = ['gpt-3.5-turbo', 'gpt-3.5-turbo-16k','gpt-4']  # Add more models if available
model_select = st.sidebar.selectbox('Select LLM Model:', llm_model_options, index=0)
st.sidebar.markdown("""\n""")
temperature_input = st.sidebar.slider('Set AI Randomness / Determinism:', min_value=0.0, max_value=1.0, value=0.0)
st.sidebar.markdown("""\n""")
clear_history = st.sidebar.button("Clear conversation history")



if "conversation" not in st.session_state:
    st.session_state.conversation = None

st.markdown(f"""## AI-Assisted Document Analysis For  Louisiana 📑 <span style=color:#2E9BF5><font size=5>Beta</font></span>""",unsafe_allow_html=True)


# Extracts and concatenates text from a list of PDF documents
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
        except (PdfReader.PdfReadError, PyPDF2.utils.PdfReadError) as e:
            print(f"Failed to read {pdf}: {e}")
            continue  # skip to next pdf document in case of read error

        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:  # checking if page_text is not None or empty string
                text += page_text
            else:
                print(f"Failed to extract text from a page in {pdf}")
    print("Retuning the Text")
    return text

# Splits a given text into smaller chunks based on specified conditions
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def load_question_ux():
    # Initialize chat history in session state for Document Analysis (doc) if not present
    if  st.session_state.conversation != None:
        if 'doc_messages' not in st.session_state or clear_history:
            # Start with first message from assistant
            st.session_state['doc_messages'] = [{"role": "assistant", "content": "Ask A Question About BEAD"}]
            st.session_state['chat_history'] = []  # Initialize chat_history as an empty list

    # Display previous chat messages
    if  st.session_state.conversation != None:
        for message in st.session_state['doc_messages']:
            with st.chat_message(message['role']):
                st.write(message['content'])
    print("Rednering the UX!")
    # If user provides input, process it
    if  st.session_state.conversation != None:
        if user_query := st.chat_input("Enter your query here"):
            if not OPENAI_API_KEY:
                st.info("Please add your OpenAI API key to continue.")
                st.stop()
            # Add user's message to chat history
            st.session_state['doc_messages'].append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.spinner("Generating response..."):
                # Check if the conversation chain is initialized
                if 'conversation' in st.session_state:
                    st.session_state['chat_history'] = st.session_state.get('chat_history', []) + [
                        {
                            "role": "user",
                            "content": user_query
                        }
                    ]
                    # Process the user's message using the conversation chain
                    result = st.session_state.conversation({
                        "question": user_query, 
                        "chat_history": st.session_state['chat_history']})
                    response = result["answer"]
                    # Append the user's question and AI's answer to chat_history
                    st.session_state['chat_history'].append({
                        "role": "assistant",
                        "content": response
                    })
                else:
                    response = "Please upload a document first to initialize the conversation chain."
                
                # Display AI's response in chat format
                with st.chat_message("assistant"):
                    st.write(response)
                # Add AI's response to doc_messages for displaying in UI
                st.session_state['doc_messages'].append({"role": "assistant", "content": response})

# Generates embeddings for given text chunks and creates a vector store using FAISS
def get_vectorstore(text_chunks):
    # embeddings = OpenAIEmbeddings()
  #  embeddings = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Initializes a conversation chain with a given vector store
def get_conversation_chain(vectorstore):
    memory = ConversationBufferWindowMemory(memory_key='chat_history', return_message=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=temperature_input, model_name=model_select),
        retriever=vectorstore.as_retriever(),
        get_chat_history=lambda h : h,
        memory=memory
    )
    return conversation_chain

def preload_all_docs_from_path(path):
    # Load all PDFs from path

    pdf_docs = []
    for file in os.listdir(path):
        if file.endswith(".pdf"):
            pdf_docs.append(os.path.join(path, file))
    # Get PDF Text
    print(pdf_docs)
    raw_text = get_pdf_text(pdf_docs)
    # st.write(raw_text)

    # Retrieve chunks from text
    text_chunks = get_text_chunks(raw_text)
    ## st.write(text_chunks)  

    # Create FAISS Vector Store of PDF Docs
    vectorstore = get_vectorstore(text_chunks)

    # Create conversation chain
    st.session_state.conversation = get_conversation_chain(vectorstore)

   

#preload_all_docs_from_path("docs/")
#st.write("Preloaded all documents from path")
st.write(" Click the Button Below to load the AI Model and Documents when you are ready to start asking questions about BEAD")
st.write(" You can select the AI Model and set the AI Randomness / Determinism in the sidebar to the left")
st.write("Once your AI is loaded , changing the Model or Temperature will require you to reload the page")
if st.button("Click Here to Start Asking Questions About BEAD"):
    with st.spinner("Processing"):
        preload_all_docs_from_path("docs/")
        st.write("Preloaded all documents from path")
load_question_ux()




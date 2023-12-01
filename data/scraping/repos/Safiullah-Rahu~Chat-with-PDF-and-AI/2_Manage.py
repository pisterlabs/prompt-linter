# Importing the required modules
import os 
import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler
from langchain.callbacks import get_openai_callback
import logging
import time
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
import PyPDF2
from io import StringIO
import pinecone 

# Setting up logging configuration
logger = logging.getLogger("AI_Chatbot")

# Setting up Streamlit page configuration
st.set_page_config(
    page_title="AI Chatbot", layout="wide", initial_sidebar_state="expanded"
)

# Getting the OpenAI API key from Streamlit Secrets
openai_api_key = st.secrets.secrets.OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = openai_api_key

# Getting the Pinecone API key and environment from Streamlit Secrets
PINECONE_API_KEY = st.secrets.secrets.PINECONE_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
PINECONE_ENV = st.secrets.secrets.PINECONE_ENV
os.environ["PINECONE_ENV"] = PINECONE_ENV

# Initialize Pinecone with API key and environment
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)





@st.cache_data
def load_docs(files):
    all_text = []
    for file_path in files:
        file_extension = os.path.splitext(file_path.name)[1]
        if file_extension == ".pdf":
            pdf_reader = PyPDF2.PdfReader(file_path)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            all_text.append(text)
        elif file_extension == ".txt":
            stringio = StringIO(file_path.getvalue().decode("utf-8"))
            text = stringio.read()
            all_text.append(text)
        else:
            st.warning('Please provide txt or pdf.', icon="⚠️")
    return all_text  

def admin(sel_ns):
    # Set the Pinecone index name
    pinecone_index = "aichat"

    # # Initialize Pinecone with API key and environment
    # pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

    # Prompt the user to upload PDF/TXT files
    st.write("Upload PDF/TXT Files:")
    uploaded_files = st.file_uploader("Upload", type=["pdf", "txt"], label_visibility="collapsed", accept_multiple_files = True)
    
    if uploaded_files is not None:
        documents = load_docs(uploaded_files)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.create_documents(documents)

        # Initialize OpenAI embeddings
        embeddings = OpenAIEmbeddings(model = 'text-embedding-ada-002')

        # Display the uploaded file content
        file_container = st.expander(f"Click here to see your uploaded content:")
        file_container.write(docs)

        # Display success message
        st.success("Document Loaded Successfully!")

        # Checkbox for the first time document upload
        first_t = st.checkbox('Uploading Document First time.')

        st.write("---")

        # Checkbox for subsequent document uploads
        second_t = st.checkbox('Uploading Document Second time and onwards...')

        if first_t:
            # Delete the existing index if it exists
            if pinecone_index in pinecone.list_indexes():
                pinecone.delete_index(pinecone_index)
            time.sleep(50)
            st.info('Initializing Document Uploading to DB...')

            # Create a new Pinecone index
            pinecone.create_index(
                    name=pinecone_index,
                    metric='cosine',
                    dimension=1536  # 1536 dim of text-embedding-ada-002
                    )
            time.sleep(80)

            # Upload documents to the Pinecone index
            vector_store = Pinecone.from_documents(docs, embeddings, index_name=pinecone_index, namespace= sel_ns)
            
            # Display success message
            st.success("Document Uploaded Successfully!")
        
        elif second_t:
            st.info('Initializing Document Uploading to DB...')

            # Upload documents to the Pinecone index
            vector_store = Pinecone.from_documents(docs, embeddings, index_name=pinecone_index, namespace= sel_ns)
            
            # Display success message
            st.success("Document Uploaded Successfully!")

pinecone_index = "aichat"
# Check if the Pinecone index exists
time.sleep(5)
if pinecone_index in pinecone.list_indexes():
    index = pinecone.Index(pinecone_index)
    index_stats_response = index.describe_index_stats()
    # Display the available documents in the index
    #st.info(f"The Documents available in index: {list(index_stats_response['namespaces'].keys())}")
    # Define the options for the dropdown list
    options = list(index_stats_response['namespaces'].keys())
st.session_state.sel_namespace = ""
# Display a text input box in the sidebar to enter the password
passw = st.sidebar.text_input("Enter your password: ", type="password")
# Call the admin() function if the correct password is entered
if passw == "ai4chat":
    #namespa = st.text_input("Enter Namespace Name: ")
    exist_name = st.checkbox('Use Existing Namespace to Upload Docs')
    del_name = st.checkbox("Delete a Namespace")
    new_name = st.checkbox("Create New Namespace to Upload Docs")
    if exist_name:
        st.write("---")
        st.write("Existing Namespaces:👇")
        st.write(options)
        # Create a dropdown list
        selected_namespace = st.text_input("Enter Existing Namespace Name: ") #st.sidebar.selectbox("Select a namespace", options)
        st.session_state.sel_namespace = selected_namespace
        st.warning("Use 'Uploading Document Second time and onwards...' button to upload docs in existing namespace!", icon="⚠️")
        #selected_namespace = selected_namespace
        # Display the selected value
        st.write("You selected:", st.session_state.sel_namespace)
    if del_name:
        st.write("---")
        st.write("Existing Namespaces:👇")
        st.write(options)
        # Create a dropdown list
        selected_namespace = st.text_input("Enter Existing Namespace Name: ") #st.sidebar.selectbox("Select a namespace", options)
        st.session_state.sel_namespace = selected_namespace
        st.warning("The namespace will be permanently deleted!", icon="⚠️")
        del_ = st.checkbox("Check this to delete Namespace")
        if del_:
            with st.spinner('Deleting Namespace...'):
                time.sleep(5)
                index.delete(namespace=st.session_state.sel_namespace, delete_all=True)
            st.success('Successfully Deleted Namespace!')
    if new_name:
        selected_namespace = st.text_input("Enter Namespace Name: (For Private Namespaces use .sec at the end, e.g., testname.sec)")
        st.session_state.sel_namespace = selected_namespace
    sel_ns = st.session_state.sel_namespace
    admin(sel_ns)
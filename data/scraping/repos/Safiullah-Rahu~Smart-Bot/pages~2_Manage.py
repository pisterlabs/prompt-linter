import streamlit as st
import os
import time
import pinecone
import PyPDF2
from io import StringIO
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import WebBaseLoader


# Setting up Streamlit page configuration
st.set_page_config(
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Getting the OpenAI API key from Streamlit Secrets
openai_api_key = st.secrets.secrets.OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = openai_api_key

# Getting the Pinecone API key and environment from Streamlit Secrets
PINECONE_API_KEY = st.secrets.secrets.PINECONE_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
PINECONE_ENV = st.secrets.secrets.PINECONE_ENV
os.environ["PINECONE_ENV"] = PINECONE_ENV

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
    # st.write(all_text)
    return all_text  

def select_index():
    pinecone_index_list = pinecone.list_indexes()
    return pinecone_index_list

def manage_chat():

    if 'clicked' not in st.session_state:
        st.session_state.clicked = False

    def click_button():
        st.session_state.clicked = True

    # Initialize Pinecone with API key and environment
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    M_OPTIONS = ["Create New Index", "Use Existing Index", "Delete Index"]
    manage_opts = st.sidebar.selectbox(label="Select Option", options=M_OPTIONS)

    if manage_opts == "Create New Index":
        st.header("Create New Index to Upload Documents")
        st.write("---")
        col1, col2 = st.columns([1,1])
        doc_ = col1.checkbox("Upload Documents [PDF/TXT]")
        url_ = col2.checkbox("Upload Website Content [URL]")
        # Prompt the user to upload PDF/TXT files
        try:
            if doc_:
                st.write("Upload PDF/TXT Files:")
                uploaded_files = st.file_uploader("Upload", type=["pdf", "txt"], label_visibility="collapsed", accept_multiple_files = True)
                if uploaded_files != []:
                    st.info('Initializing Document Loading...')
                    documents = load_docs(uploaded_files)
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
                    docs = text_splitter.create_documents(documents)
                    st.success("Document Loaded Successfully!")
            elif url_:
                #web_list = []
                website_ = st.text_input("Enter website URL:")
                if website_ != "":
                    st.info('Initializing Website Loading...')
                    loader = WebBaseLoader(website_)
                    loader.requests_kwargs = {'verify':False}
                    docs = loader.load()
                    st.success('Website Successfully Loaded!')



            # Initialize OpenAI embeddings
            embeddings = OpenAIEmbeddings(model = 'text-embedding-ada-002')

            # Display the uploaded file content
            file_container = st.expander(f"Click here to see your uploaded content:")
            file_container.write(docs)

            # Display success message
            # st.success("Document Loaded Successfully!")
            pinecone_index = st.text_input("Enter the name of Index: ")
            if pinecone_index != "":
                st.info('Initializing Index Creation...')
                # Create a new Pinecone index
                pinecone.create_index(
                        name=pinecone_index,
                        metric='cosine',
                        dimension=1536  # 1536 dim of text-embedding-ada-002
                        )
                st.success('Index Successfully Created!')
                time.sleep(80)
                st.info('Initializing Document Uploading to DB...')
                # Upload documents to the Pinecone index
                vector_store = Pinecone.from_documents(docs, embeddings, index_name=pinecone_index)
                
                # Display success message
                st.success("Document Uploaded Successfully!")
        except:
            st.toast("Select any one option to proceed.")

    elif manage_opts == "Use Existing Index":
        st.header("Use Existing Index to Upload Documents")
        st.write("---")
        pinecone_index_list = select_index()
        pinecone_index = st.selectbox(label="Select Index", options = pinecone_index_list )
        #col1, col2 = st.columns([1,1])
        # doc_ = col1.checkbox("Upload Documents [PDF/TXT]")
        # url_ = col2.checkbox("Upload Website Content [URL]")
        doc_url = st.radio(
            "Select Content Format",
            ["Upload Documents [PDF/TXT]", "Upload Website Content [URL]"],
            key="visibility",
            horizontal=True,
            disabled=False
            )
        # Prompt the user to upload PDF/TXT files
        try:
            if doc_url == "Upload Documents [PDF/TXT]":
                st.write("Upload PDF/TXT Files:")
                uploaded_files = st.file_uploader("Upload", type=["pdf", "txt"], label_visibility="collapsed", accept_multiple_files = True)
                if uploaded_files != []:
                    st.info('Initializing Document Loading...')
                    documents = load_docs(uploaded_files)
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
                    docs = text_splitter.create_documents(documents)
                    st.success("Document Loaded Successfully!")

            elif doc_url == "Upload Website Content [URL]":
                #web_list = []
                website_ = st.text_input("Enter website URL:")
                if website_ != "":
                    st.info('Initializing Website Loading...')
                    loader = WebBaseLoader(website_)
                    loader.requests_kwargs = {'verify':False}
                    docs = loader.load()
                    st.success('Website Successfully Loaded!')

            # Initialize OpenAI embeddings
            embeddings = OpenAIEmbeddings(model = 'text-embedding-ada-002')
            # Display success message
            #st.success("Document Loaded Successfully!")
            # Display the uploaded file content
            file_container = st.expander(f"Click here to see your uploaded content:")
            file_container.write(docs)
            up_check = st.checkbox('Check this to Upload Docs in Selected Index')
            if up_check:
                st.info('Initializing Document Uploading to DB...')
                # Upload documents to the Pinecone index
                time.sleep(40)
                vector_store = Pinecone.from_documents(docs, embeddings, index_name=pinecone_index)
                
                # Display success message
                st.success("Document Uploaded Successfully!")
        except:
            st.toast("Select any one option to proceed.")

    elif manage_opts == "Delete Index":
        st.header("Delete Any Existing Index")
        st.write("---")
        pinecone_index_list = select_index()
        pinecone_index = st.selectbox(label="Select Index", options = pinecone_index_list )
        # time.sleep(10)
        # st.write("Existing Indexes:👇")
        # st.write(pinecone.list_indexes())
        # pinecone_index = st.text_input("Write Name of Existing Index to delete: ")
        st.write(f"The Index named '{pinecone_index}' is selected for deletion.")
        del_check = st.checkbox('Check this to Delete Index')
        if del_check:
            with st.spinner('Deleting Index...'):
                time.sleep(5)
                pinecone.delete_index(pinecone_index)
                time.sleep(10)
            st.success(f"'{pinecone_index}' Index Deleted Successfully!")

man_pass = "chatadmin"
pass_manage = st.sidebar.text_input("Enter Password: ", type="password")
if pass_manage == man_pass:
    manage_chat()
else:
    st.sidebar.warning("Incorrect Password!")

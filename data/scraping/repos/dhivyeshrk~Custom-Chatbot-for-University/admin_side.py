# admin_app.py
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import streamlit as st
import os
import platform
import numpy as np


# Set Streamlit page configuration
st.set_page_config(
    page_title='Admin - LLM QA File',
    page_icon=":information_desk_person:",
    menu_items=None
)

# Define Razer-themed background styling
st.markdown(
    """
    <style>
    body {
        background-image: url('https://your-image-url-here.com');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# loading PDF, DOCX and TXT files as LangChain Documents
def load_document(file):
    import os
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        from langchain.document_loaders import TextLoader
        loader = TextLoader(file)
    else:
        print('Document format is not supported!')
        return None

    data = loader.load()
    return data


# splitting data i  n chunks
def chunk_data(data, chunk_size=256, chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks


# create embeddings using OpenAIEmbeddings() and save them in a Chroma vector store
def create_embeddings(chunks):
    embeddings = OpenAIEmbeddings()
    # vector_store = Chroma.from_documents(chunks, embeddings)
    en = np.shape(chunks)[0]
    ids = np.full((en,), "Hello")
    vector_store = Chroma.from_documents(chunks, embeddings,
                                         persist_directory=r"C:\Users\DELL\DataspellProjects\Chatbot_Test_2\ChatWithDocument\Data_Files_2",
                                         collection_metadata = {"purpose": "To store documents about people"})

    # for i, chunk in enumerate(chunks):
    #     vector_store.update(
    #         {"text": f"Dhivyesh {chunk}", "embedding": embeddings[i]},
    #         metadata={"author": "Dhivyesh"},
    #     )
    return vector_store
def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']
def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    
    return total_tokens, (total_tokens / 1000 * 0.0001 * 18.76) 

# Main Streamlit app for admin side
if __name__ == "__main__":
    st.subheader('Admin - LLM Document Upload and Processing :information_desk_person:')
   

    # loading the OpenAI api key from .env
    # from dotenv import load_dotenv, find_dotenv
    # load_dotenv(find_dotenv(), override=True)
    os.environ['OPENAI_API_KEY'] = ''
    st.subheader('LLM Question-Answering Application :information_desk_person:')

    if 'mobile' in platform.platform().lower():
        print('Click here to enter file ')

    #with st.sidebar:


    with st.expander("Load file"):
        # file uploader widget
        uploaded_file = st.file_uploader('Upload a file:', type=['pdf', 'docx', 'txt'])        
        st.markdown('<p font-size:10px>(Any file Any Format)</p>', unsafe_allow_html=True)
        # chunk size number widget
        chunk_size = st.number_input('Chunk size:', min_value=100, max_value=2048, value=512, on_change=None)
        # k number input widget
        k = st.number_input('k', min_value=1, max_value=20, value=3, on_change=clear_history)
        # add data button widget
        add_data = st.button('Add Data', on_click=clear_history)
        expanded=True
      
        
        if uploaded_file and add_data: # if the user browsed a file
            with st.spinner('Reading, chunking and embedding file ...'):

                # writing the file from RAM to the current directory on disk
                bytes_data = uploaded_file.read()
                file_name = os.path.join('./', uploaded_file.name)
                with open(file_name, 'wb') as f:
                    f.write(bytes_data)

                data = load_document(file_name)
                chunks = chunk_data(data, chunk_size=chunk_size)
                #st.write(f'Chunk size: {chunk_size}, Chunks: {len(chunks)}')

                tokens, embedding_cost = calculate_embedding_cost(chunks)
                #st.write(f'Embedding cost: R{embedding_cost:}')

                # creating the embeddings and returning the Chroma vector store
                vector_store = create_embeddings(chunks)

                # saving the vector store in the streamlit session state (to be persistent between reruns)
                st.session_state.vs = vector_store
                st.success('File uploaded, chunked and embedded successfully.')


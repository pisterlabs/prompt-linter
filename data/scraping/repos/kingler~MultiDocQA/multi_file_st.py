import os
import re
import io
import contextlib
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.router import MultiRetrievalQAChain
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
import shutil


def main():
    st.title(':blue[QA over documents with langchain router chain]')

    # Check for OpenAI API Key in environment variable
    st.sidebar.header('API Key')
    if 'OPENAI_API_KEY' not in os.environ:
        openai_api_key = st.sidebar.text_input('Enter your OpenAI API Key', type='password')
        if openai_api_key:
            
            os.environ['OPENAI_API_KEY'] = openai_api_key
    else:
        st.sidebar.write(":green[API Key set successfully.]")

    # Initialize the OpenAI embeddings
    embedding = OpenAIEmbeddings()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20, length_function=len)

    # Initialize session_state
    if "retrievers" not in st.session_state:
        st.session_state.retrievers = []
    if "retriever_descriptions" not in st.session_state:
        st.session_state.retriever_descriptions = []
    if "retriever_names" not in st.session_state:
        st.session_state.retriever_names = []

    # Directories for storing indexes and uploaded files
    indexes_dir = 'indexes'
    docs_dir = 'docs'
    os.makedirs(indexes_dir, exist_ok=True)
    os.makedirs(docs_dir, exist_ok=True)

    if "initialized" not in st.session_state:
        st.session_state.initialized = False

    indexes = [f for f in os.listdir(indexes_dir) if os.path.isdir(os.path.join(indexes_dir, f))]
    if not st.session_state.initialized:
        # Process existing indexes
        
        for index in indexes:
            if index not in st.session_state.retriever_names:
                retriever = Chroma(persist_directory=os.path.join(indexes_dir, index), embedding_function=embedding).as_retriever()
                st.session_state.retrievers.append(retriever)
                st.session_state.retriever_names.append(index)
                st.session_state.retriever_descriptions.append(f"Good for answering questions about {index}")

        st.session_state.initialized = True

    st.sidebar.header('Uploaded Files')
    uploaded_files = [f for f in os.listdir(docs_dir) if os.path.isfile(os.path.join(docs_dir, f))]
    st.sidebar.write(uploaded_files)

    st.sidebar.header('Document Indexes')
    st.sidebar.write(indexes)

    # Save uploaded files to "docs" folder
    files = st.file_uploader('Upload files', type=['txt', 'pdf'], accept_multiple_files=True)
    if files:
        st.session_state.files = files  # Save uploaded files to session state
        for file in files:
            filename = file.name
            filepath = os.path.join(docs_dir, filename)
            with open(filepath, "wb") as f:
                f.write(file.getvalue())

    # Check for each file in the "docs" folder and create/load the index
    for filename in os.listdir(docs_dir):
        filepath = os.path.join(docs_dir, filename)
        if os.path.exists(os.path.join(indexes_dir, filename[:-4])):
            continue
        else:
            with st.spinner(f'Creating index for {filename}...'):
                if filename.endswith('.txt'):
                    with open(filepath, 'r', encoding="utf-8", errors="ignore") as f:
                        doc = text_splitter.create_documents([f.read()])
                elif filename.endswith('.pdf'):
                    with open(filepath, 'rb') as f:
                        loader = PyPDFLoader(filepath)
                        doc = loader.load_and_split()

                if doc is not None:
                    retriever = Chroma.from_documents(documents=doc, embedding=embedding, persist_directory=os.path.join(indexes_dir, filename[:-4]))
                    retriever.persist()
                    st.session_state.retrievers.append(retriever)
                    st.session_state.retriever_names.append(filename[:-4])
                    st.session_state.retriever_descriptions.append(f"Good for answering questions about {filename[:-4]}")
                    st.success(f'Index created for {filename}.')

    # st.write(st.session_state.retrievers)
    chain = MultiRetrievalQAChain.from_retrievers(OpenAI(), st.session_state.retriever_names, st.session_state.retriever_descriptions, st.session_state.retrievers, verbose=True)

    st.header('Ask a Question')
    question = st.text_input('Enter your question')
    if st.button('Ask'):

        if not st.session_state.retrievers:
            st.warning("Please upload files or ensure they have been indexed.")
            return

        with st.spinner('Processing your question...'):           

            with io.StringIO() as buf, contextlib.redirect_stdout(buf):
                # st.write("INDEXES: ", st.session_state.retrievers)

                resp = chain.run(question)
                output = buf.getvalue()

            match = re.search(r"(\w+: \{'query': '.*?'\})", output)
            if match:
                # write wihch index we are using in green
                st.write(":green[We are using the following index:]")
                st.write(match.group(1))
            else:
                st.write("No match found.")
            st.write(":green[Answer:]")
            st.write(resp)


if __name__ == '__main__':
    main()

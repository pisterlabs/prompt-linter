import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from pathlib import Path
import os
# --- GENERAL SETTINGS ---
PAGE_TITLE = "DENSO GPT Expert"
PAGE_ICON = "🤖"
AI_MODEL_OPTIONS = [
    "gpt-3.5-turbo",
    "gpt-4",
    "gpt-4-32k",
]

loaders = []
docs = []

current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
storage_dir = current_dir.parent.parent / "storage"
persist_directory_dir = str(current_dir.parent.parent.joinpath("chroma_db"))
model = HuggingFaceEmbeddings(model_name="bkai-foundation-models/vietnamese-bi-encoder")
database = Chroma(persist_directory=persist_directory_dir, embedding_function=model)

ss = st.session_state
ss.setdefault('debug', {})

def split_documents_into_chunks(docs, chunk_size=800, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # Splitting the documents into chunks
    chunks = text_splitter.split_documents(docs)

    # Create a Chroma collection from the document chunks
    Chroma.from_documents(documents=chunks, embedding=model, persist_directory=persist_directory_dir)

    return chunks

if __name__ == '__main__':
    st.title(PAGE_TITLE)
    st.sidebar.title(f"{PAGE_ICON} {PAGE_TITLE}")
    st.sidebar.write("Welcome to the DENSO GPT Expert")

    st.write('## Upload your PDF')
    t1, t2 = st.tabs(['UPLOAD', 'STORAGE'])
    with t1:
        uploaded_files = st.file_uploader("Choose a PDF file", type='pdf', accept_multiple_files=True)
        result = st.button("Save to storage")
        # if the user clicks the button
        if len(uploaded_files) and result:
            with st.spinner("Processing..."):
                if not storage_dir.exists():
                    storage_dir.mkdir(parents=True, exist_ok=True)
                for uploaded_file in uploaded_files:
                    # save the uploaded file to the storage folder
                    with open(os.path.join(storage_dir, uploaded_file.name), "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.success("Saved File:{} to storage".format(uploaded_file.name))

                files_in_storage = list(storage_dir.iterdir())
                if len(files_in_storage) > 0:
                    for file in files_in_storage:
                        file_path = str(storage_dir / file.name)
                        loader = PyPDFLoader(file_path=file_path)
                        loaders.append(loader)
                    for loader in loaders:
                        docs.extend(loader.load())
                # Split the documents into chunks
                chunks = split_documents_into_chunks(docs=docs, chunk_size=800, chunk_overlap=20)
                st.success("Split documents into chunks")

    with t2:
        if storage_dir.exists():
            files_in_storage = list(storage_dir.iterdir())
            if files_in_storage:
                st.write("### Files in storage:")
                for file in files_in_storage:
                    st.write(file.name)
            else:
                st.write("No files in storage.")
        else:
            st.write("Storage folder does not exist.")

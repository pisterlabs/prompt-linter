import streamlit as st
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings

from src.utils import get_config, vectordb, read_markdown_file

# Import config vars
cfg = get_config()

# Load existing vectordb
loaded_vectordb = vectordb(cfg)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=cfg.CHUNK_SIZE, chunk_overlap=cfg.CHUNK_OVERLAP
)

embeddings = HuggingFaceEmbeddings(
    model_name=cfg.EMBEDDINGS_MODEL_NAME,
    model_kwargs={"device": "cuda" if cfg.USE_GPU else "cpu"},
)


st.set_page_config(
    page_title="Ask Canonical - Embeddings",
    page_icon="https://assets.ubuntu.com/v1/49a1a858-favicon-32x32.png",
    layout="wide",
)
st.title(":penguin: Ask Canonical - Embeddings", anchor=False)


with st.sidebar:
    st.markdown(read_markdown_file("media/embeddings-documentation.md"))

uploaded_files = st.file_uploader(
    "Upload additional documentation",
    type=("txt", "md"),
    accept_multiple_files=True,
    help="Add new files for embeddings",
)

if uploaded_files:
    texts = [uploaded_file.read().decode("utf-8") for uploaded_file in uploaded_files]

    new_docs_vectorstore = FAISS.from_texts(texts, embeddings)
    loaded_vectordb.merge_from(new_docs_vectorstore)

    loaded_vectordb.save_local(cfg.DB_FAISS_PATH)
    st.success(
        f"{len(uploaded_files)} files processed and saved in the vector store.",
        icon="✅",
    )

import streamlit as st

from langchain.document_loaders import Docx2txtLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os
from langchain.document_loaders import PyPDFLoader


api = os.environ.get("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(openai_api_key=api)
uploaded_file = st.file_uploader("**Upload Your txt File**")
if uploaded_file:
    xy = os.getcwd()
    upload_dir = f"{xy}/uploads/"
    os.makedirs(upload_dir, exist_ok=True)  # Create the directory if it doesn't exist

    file_path = os.path.join(upload_dir, uploaded_file.name)

    #file_path = "./uploads/" + uploaded_file.name  # Specify the desired directory
    with open(file_path, "wb") as file:
        file.write(uploaded_file.read())
    st.write("File saved to:", file_path)

    st.write(uploaded_file)
    st.download_button("Download PDF", data=uploaded_file, file_name=uploaded_file.name)
    lala =[]

    loader = PyPDFLoader(f"{file_path}")
    #loader = Docx2txtLoader(f"{file_path}")

    data = loader.load()
    #print(data)
    lala.extend(data)
    st.write(lala)
    """

    loader = PyPDFLoader("/Users/poojas/Downloads/BCG.pdf")
    pages = loader.load()
    st.write(pages)
    lala.extend(pages)
    """

    with st.spinner("It's indexing..."):
        index = (FAISS.from_documents(lala, embeddings))
    st.success("Embeddings done.", icon="✅")


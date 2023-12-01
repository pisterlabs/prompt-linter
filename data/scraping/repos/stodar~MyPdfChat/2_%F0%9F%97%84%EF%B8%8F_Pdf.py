import streamlit as st
from PyPDF2 import PdfReader
import os
from langchain.indexes import VectorstoreIndexCreator

with open('ApiKey.txt', 'r') as file:
    my_code = file.read()

os.environ["OPENAI_API_KEY"] = str(my_code)

st.title("PDF")

if "my_input" not in st.session_state:
    st.session_state["my_input"] = ""

my_pdf = st.file_uploader("Drop PDF here", "pdf", False)
submit = st.button("Submit")




def pdftotext(mypdf):
    reader = PdfReader(mypdf)
    pages = [page.extract_text() for page in reader.pages]
    text = " ".join(pages)
    with open('./data/data.txt', 'w') as f:
        f.write(text)
    from langchain.document_loaders import TextLoader
    loader = TextLoader("./data/data.txt")
    index = VectorstoreIndexCreator().from_loaders([loader])
    return index


if submit:
    st.session_state["my_input"] = pdftotext(my_pdf)
    st.success("Go to Chat Page")
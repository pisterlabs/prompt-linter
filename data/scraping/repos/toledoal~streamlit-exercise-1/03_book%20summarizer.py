# Path: pages/03_book summarizer.py
import streamlit as st

from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document

from langchain import OpenAI
from langchain.text_splitter import CharacterTextSplitter

# import firebase_admin
# from firebase_admin import credentials
# from firebase_admin import firestore

# cred = credentials.Certificate("./firebase-key.json")
# firebase_admin.initialize_app(cred)

# db = firestore.client()


# def getDocuments():
#     doc_ref = db.collection("books").document("FDQL8EeCW32F2aXG104O")

#     doc = doc_ref.get()
#     if doc.exists:
#         st.write(f"Document data: {doc.to_dict()}")
#     else:
#         st.write("No such document!")


def load_LLM():
    """Logic for loading the chain you want to use should go here."""
    # Make sure your openai_api_key is set as an environment variable
    llm = OpenAI()
    return llm


def getFile(file):
    return file.getvalue().decode("utf-8").splitlines()


def processText():
    if file is not None:
        text_splitter = CharacterTextSplitter(" ")
        sd = str(getFile(file))
        texts = text_splitter.split_text(sd)
        return texts
    else:
        return []


def run():
    docs = [Document(page_content=t) for t in processText()[:3]]
    # key = st.secrets["OPENAI_KEY"]
    chain = load_summarize_chain(load_LLM(), chain_type="map_reduce", verbose=True)
    return chain.run(docs)


st.title("Book summarizer")

st.write("## Summarize this book")

# getDocuments()


file = st.file_uploader("Upload a file", type="txt")


st.session_state["file"] = file

numberOfDocuments = 0
fileName = ""
if st.button("Process book into documents", disabled=(file is None)):
    st.write("Number of small document chunks:")
    numberOfDocuments = st.write(len(processText()))
    fileName = st.write(file.name)

resultOfSummarization = ""
if st.button("Run", disabled=(file is None or numberOfDocuments == 0)):
    if file is not None:
        resultOfSummarization = run()


message = f"Summarize: {file.name}" if file is not None else "No file selected"
st.write(message)
st.write(resultOfSummarization)

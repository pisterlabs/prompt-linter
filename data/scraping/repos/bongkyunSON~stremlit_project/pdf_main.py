# __import__("pysqlite3")
# import sys
# sys.modules["sqlite3"] = sys.modules.pop["pysqlite3"]

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
import streamlit as st
import tempfile
import os

# 제목
st.title("ChatPDF")
st.write("-"*30)

#파일 업로드
uploaded_file = st.file_uploader("PDF 업로드 해주세요", type=["pdf"])
st.write("-"*30)

def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_file_path = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    loader = PyPDFLoader(temp_file_path)
    pages = loader.load_and_split()
    return pages


# 업로드 되면 동작하는 코드
if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)

    # Split
    text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 300,
    chunk_overlap  = 20,
    length_function = len,
    is_separator_regex = False,
    )

    texts = text_splitter.split_documents(pages)

    # Embedding
    embeddings_model = OpenAIEmbeddings()

    # load it into Chroma
    db = Chroma.from_documents(texts, embeddings_model)

    #Qustion
    st.header("PDF에게 질문해보세요")
    query = st.chat_input("질문을 입력해주세요")
    if query:
        with st.spinner('잠시만 기다려 주세요...'):
            llm = ChatOpenAI(model_name = "gpt-3.5-turbo", temperature=0)
            qa_chain = load_qa_chain(llm, chain_type="stuff")
            qa = RetrievalQA(combine_documents_chain=qa_chain, retriever=db.as_retriever())
            result = qa.run(query)
            st.write(result)

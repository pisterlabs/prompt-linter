from dotenv import load_dotenv
import os
import pandas as pd
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import AzureChatOpenAI
import openai


def main():
    st.set_page_config(page_title="Megazone Cloud ChatBot")
    st.markdown("<h1 style='text-align: center; color: lightgreen;'>Megazone Cloud ChatBot 💬</h1>",
                unsafe_allow_html=True)

    # load environment variables
    load_dotenv()

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
    OPENAI_DEPLOYMENT_NAME = os.getenv("OPENAI_DEPLOYMENT_NAME")
    OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")
    OPENAI_EMBEDDING_DEPLOYMENT_NAME = os.getenv(
        "OPENAI_EMBEDDING_DEPLOYMENT_NAME")
    OPENAI_EMBEDDING_MODEL_NAME = os.getenv("OPENAI_EMBEDDING_MODEL_NAME")
    OPENAI_DEPLOYMENT_VERSION = os.getenv("OPENAI_DEPLOYMENT_VERSION")

    # init Azure OpenAI
    openai.api_type = "azure"
    openai.api_version = OPENAI_DEPLOYMENT_VERSION
    openai.api_base = OPENAI_API_BASE
    openai.api_key = OPENAI_API_KEY

    # init openai
    llm = AzureChatOpenAI(deployment_name=OPENAI_DEPLOYMENT_NAME,
                          model_name=OPENAI_MODEL_NAME,
                          openai_api_base=OPENAI_API_BASE,
                          openai_api_version=OPENAI_DEPLOYMENT_VERSION,
                          openai_api_key=OPENAI_API_KEY)

    embeddings = OpenAIEmbeddings(
        deployment=OPENAI_EMBEDDING_DEPLOYMENT_NAME, model=OPENAI_EMBEDDING_MODEL_NAME, chunk_size=1)

    # upload file
    uploaded_file = st.file_uploader("Upload your file", type=[
                                     "pdf", "csv", "txt", "xlsx", "xls"])

    # extract the text
    if uploaded_file is not None:
        file_details = {"FileName": uploaded_file.name,
                        "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
        st.write(file_details)

        if file_details["FileType"] == "application/pdf":
            with st.spinner('Reading the PDF...'):
                pdf_reader = PdfReader(uploaded_file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()

        elif file_details["FileType"] == "text/plain":
            with st.spinner('Reading the TXT file...'):
                text = uploaded_file.read().decode("utf-8")

        elif file_details["FileType"] in ["application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
            with st.spinner('Reading the Excel file...'):
                df = pd.read_excel(uploaded_file)
                text = " ".join(map(str, df.values))

        elif file_details["FileType"] == "text/csv":
            with st.spinner('Reading the CSV file...'):
                df = pd.read_csv(uploaded_file)
                text = " ".join(map(str, df.values))

        else:
            st.error("File type not supported.")

        # split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
        chunks = text_splitter.split_text(text)

        # load the faiss vector store we saved into memory
        with st.spinner('Creating knowledge base...'):
            vectorStore = FAISS.from_texts(chunks, embeddings)

        # use the faiss vector store we saved to search the local document
        retriever = vectorStore.as_retriever(
            search_type="similarity", search_kwargs={"k": 2})

        # use the vector store as a retriever
        qa = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=False)

        # show user input
        user_question = st.text_input("Ask a question 🤖:")
        if user_question:
            result = qa({"query": user_question})
            # Display the result in a more noticeable way
            st.markdown(
                f'### Answer: \n {result["result"]}', unsafe_allow_html=True)


if __name__ == '__main__':
    main()

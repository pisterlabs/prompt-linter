import streamlit as st
from langchain.document_loaders import UnstructuredExcelLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI


def function_for_question(name):
    print(name)


def UploadedReport():
    st.write("Please upload your Weekly Report Here")

    data = st.file_uploader("Upload a Report")
    if data:
        index_creator = VectorstoreIndexCreator()
        malwareEventsLoader = UnstructuredExcelLoader(
            "/var/www/html/serianu_projects/projects/LangChain_StreamLit/Langchain/Data/malwareEvents.xlsx")
        malwareEventsDocsearch = index_creator.from_loaders([malwareEventsLoader])
        malwareEventsChain = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff",
                                                         retriever=malwareEventsDocsearch.vectorstore.as_retriever(),
                                                         input_key="question")

    query = st.text_area("Insert your query")

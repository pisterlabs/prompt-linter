import openai
import streamlit as st
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
import os
OPENAI_API_KEY='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'




import gradio as gr

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os



def process_file(file_up,chunk_size,chunk_overlap):
    loader = UnstructuredFileLoader(file_up)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
    all_splits = text_splitter.split_documents(data)

    vectorstore = FAISS.from_documents(all_splits, OpenAIEmbeddings())
    return vectorstore

def query_chain(vectorstore,llm , question,temperature):
    template = """Use the following pieces of context to answer the question at the end. 
                    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
                    Use required sentences maximum and keep the answer concise. 
                    {context}
                    Question: {question}
                    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template,)

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=temperature)
    qa_chain = RetrievalQA.from_chain_type(llm,
                                           retriever=vectorstore.as_retriever(),
                                           chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

    result = qa_chain({"query": question})
    return result["result"]

    

def main():
    st.set_page_config(layout="wide")
    st.title("Talk to your TextFiles")
    llm =st. sidebar.selectbox("LLM",["GPT-3.5","GPT4","Default"])
    chunk_size = st.sidebar.slider("Chunk Size", min_value=10, max_value=10000,
                                   step=10, value=500)
    chunk_overlap=st.sidebar.slider("Chunk Overlap", min_value=5, max_value=5000,
                                    step=10, value=0)
    file_up= st.text_input("Enter the File Path (i.e ./name.format)")
    st.write("Note: Fle should be in directory of Project")
    question = st.text_input("Enter your Question")
    st.spinner("In progress...")
    
    tempature= st.sidebar.number_input("Set the ChatGPT Temperature",
                                       min_value=0.0,
                                       max_value=1.0,
                                       step=0.1,
                                       value=0.5)

    
    if file_up!="":
        vectorstore=process_file(file_up,chunk_size,chunk_overlap)
        st.write("File loaded sucessfully")
    if llm=="GPT-3.5-turbo":
        llm=ChatOpenAI(model_name="gpt-3.5-turbo",temperature=tempature)
    elif llm=="GPT4":
        llm=ChatOpenAI(model_name="gpt-4",temperature=tempature)
    else:
        llm=ChatOpenAI(model_name="gpt-3.5-turbo",temperature=tempature)
    
    if st.button("Submitt"):
        result=query_chain(vectorstore, llm,question,tempature)
        st.write("Results : ",result)

    



if __name__=="__main__":
    main()

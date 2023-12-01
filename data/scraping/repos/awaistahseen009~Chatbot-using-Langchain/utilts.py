from PyPDF2 import PdfReader
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory,ChatMessageHistory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
import streamlit as st
import pandas as pd
import io
from langchain.agents import initialize_agent , AgentType ,load_tools
def get_docs(docs,label):
    all_text=""
    if label=='PDF':
        for doc in docs:
            reader=PdfReader(doc)
            for page in reader.pages:
                all_text+=page.extract_text()
    elif label == 'CSV':
        for doc in docs:
            # Read the contents of the CSV file as bytes
            csv_contents = doc.read()

            try:
                # Try reading with utf-8 encoding
                df = pd.read_csv(io.BytesIO(csv_contents), encoding="utf-8")
            except UnicodeDecodeError:
                # If utf-8 decoding fails, try a different encoding (e.g., latin-1)
                df = pd.read_csv(io.BytesIO(csv_contents), encoding="latin-1")

            # Process the CSV data (you can access DataFrame 'df' here)
            # For example, you can add the CSV data to 'all_text' if needed
            all_text += df.to_string() 

    return all_text

def get_doc_chunks(doc):
    splitter=CharacterTextSplitter(separator='\n',chunk_size=500,chunk_overlap=150,length_function=len)
    text_chunks=splitter.split_text(doc)
    return text_chunks

def get_embedding_store(chunks):
    embedding_model=OpenAIEmbeddings()
    embedding_store=FAISS.from_texts(texts=chunks,embedding=embedding_model)
    return embedding_store

def get_conversation_chain(embedding_store):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversational_chain = ConversationalRetrievalChain.from_llm(llm=llm,retriever=embedding_store.as_retriever(),memory=memory)
    return conversational_chain


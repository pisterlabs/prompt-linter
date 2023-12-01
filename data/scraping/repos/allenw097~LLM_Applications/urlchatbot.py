#pip install streamlit langchain openai faiss-cpu tiktoken

import streamlit as st
import os
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import WebBaseLoader
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

import tempfile

article_url = st.sidebar.text_input(
    label="#### Your Web Article URL 👇",
    placeholder="Paste your article link")


os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

if article_url:
    loader = WebBaseLoader(article_url)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 0)
    splits = text_splitter.split_documents(loader.load())
    vectors = Chroma.from_documents(documents=splits,embedding=OpenAIEmbeddings())
    chain = ConversationalRetrievalChain.from_llm(llm = ChatOpenAI(temperature=0.2,model_name='gpt-3.5-turbo'),
                                                                      retriever=vectors.as_retriever())

    def conversational_chat(query):
        
        result = chain({"question": query, "chat_history": st.session_state['history']})
        st.session_state['history'].append((query, result["answer"]))
        
        return result["answer"]
    
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello ! Ask me anything about " + article_url + " 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey ! 👋"]
        
    #container for the chat history
    response_container = st.container()
    #container for the user's text input
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            
            user_input = st.text_input("Query:", placeholder="Talk about your article here (:", key='input')
            submit_button = st.form_submit_button(label='Send')
            
        if submit_button and user_input:
            output = conversational_chat(user_input)
            
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
                

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
# import openai
from io import StringIO
from langchain.chat_models import ChatOpenAI
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
# from langchain.chains import ConversationalRetrievalChain
from langchain.chains.summarize import load_summarize_chain
import tempfile
if "file_uploader_key" not in st.session_state:
    st.session_state["file_uploader_key"] = 0

if "uploaded_files" not in st.session_state:
    st.session_state["uploaded_files"] = []

# Prompt Template
template = """You are a chatbot having a conversation with a human.

Given the following extracted parts of a long document and a question, create a final answer.

{context}

{chat_history}
Human: {human_input}
Chatbot:"""

# Init Prompt
prompt = PromptTemplate(
    input_variables=["chat_history", "human_input", "context"], template=template
)

a = st.container()
with a:
    st.title("CHATBOT")
    global openai_api_key
    openai_api_key = st.text_input('OpenAI API Key', type='password')
if openai_api_key:
    @st.cache_resource
    def llm():
        model = OpenAI(temperature=0.0, openai_api_key=openai_api_key)
        embedding=OpenAIEmbeddings(openai_api_key=openai_api_key)
        return model, embedding

    llm,embedding = llm()

    @st.cache_resource
    def chain():
        global memory
        memory = ConversationBufferWindowMemory(memory_key="chat_history", input_key="human_input", return_messages=True, k=3)
        chain = LLMChain(
            llm=llm, prompt=prompt, memory=memory
        )
        
        return chain

    global llm_chain
    llm_chain = chain()


    summarize_template = """Write a concise summary of the given documents:
    {text}"""
    summarize_PROMPT = PromptTemplate(template=summarize_template, input_variables=["text"])
    llm_summarize = load_summarize_chain(llm=llm, chain_type="map_reduce",  map_prompt=summarize_PROMPT)
    # chain({"input_documents": docs}, return_only_outputs=True)
    # llm_summarize = load_summarize_chain(llm, chain_type="map_reduce")


    ########################################
    ####### CHATBOT interface#############
    ########################################
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # Display chat messages from history on app rerun
    with a:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    global documents
    documents = []



    with st.sidebar:
        uploaded_files = st.file_uploader("Upload file", accept_multiple_files=True, 
                                        key=st.session_state["file_uploader_key"],
                                        type=['txt', 'pdf']
                                        #   on_change = check
                                        )

    if uploaded_files:
            # files = set([file.name for file in uploaded_files])
            st.session_state["uploaded_files"] = uploaded_files
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000 , chunk_overlap=10, separators=[" ", ",", "\n"])
            for file in uploaded_files:
                if file.name.endswith(".pdf"):
                                    # Save the uploaded file to a temporary location
                    temp_file_path = os.path.join('docs', file.name)
                    with open(temp_file_path, "wb") as temp_file:
                        temp_file.write(file.read())
                    loader = PyPDFLoader(temp_file_path)
                    # loader = loader.load()
                elif file.name.endswith('.txt'):
                    # To read file as bytes:
                    bytes_data = file.getvalue()
                    # To convert to a string based IO:
                    stringio = StringIO(file.getvalue().decode("utf-8"))
                    # To read file as string:
                    loader = stringio.read()
                    filename = os.path.join("docs",'text.txt')
                    # filename = 'docs/text.txt'
                    with open(filename,"wb") as f:
                            f.write(file.getbuffer())
                    loader = TextLoader(filename, autodetect_encoding=True)
                loader = loader.load()
                documents.extend(loader)
            documents = text_splitter.split_documents(documents)

            # Embedding
            global docsearch
            docsearch = Chroma.from_documents(documents,
                                            embedding=embedding)

    ########################################
    ########## SIDEBAR ###############
    ########################################

    # create a function that sets the value in state back to an empty list
    def clear_msg():
        st.session_state.messages = []
        llm_chain = chain()
        st.session_state["file_uploader_key"] += 1
        st.experimental_rerun()

    if uploaded_files:
        if st.sidebar.button('Summarize'):
            with a:
                query = 'Summarize uploaded documents'
                # st.chat_message("user").markdown(query)
                llm_chain.memory.chat_memory.add_user_message(query)
                # Add user message to chat history
                # st.session_state.messages.append({"role": "user", "content": query})
                response = llm_summarize.run(documents)
                # chain({"input_documents": docs}, return_only_outputs=True)

                with st.chat_message("assistant"):
                    st.markdown('SUMMARIZE:'+response)
                llm_chain.memory.chat_memory.add_ai_message(response)
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": 'SUMMARIZE:'+response})


    st.sidebar.button("Clear", on_click=clear_msg)

    ########################################
    ####### React to user input#############
    ########################################

    with a:
        if query := st.chat_input():
            # Display user message in chat message container
            st.chat_message("user").markdown(query)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": query})
            if documents:
                docs = docsearch.similarity_search(query)
            else:
                docs = 'No Context provide'
            response = llm_chain.run({"context": docs, "human_input": query})
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(response)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

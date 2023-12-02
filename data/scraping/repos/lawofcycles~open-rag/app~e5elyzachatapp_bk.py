# Internal usage
from time import  sleep
#### IMPORTS FOR AI PIPELINES 
import requests 
import streamlit as st

st.session_state['source'] = None

# #AVATARS
# av_us = './man.png'  #"🦖"  #A single emoji, e.g. "🧑‍💻", "🤖", "🦖". Shortcodes are not supported.
# av_ass = './lamini.png'

# FUNCTION TO LOG ALL CHAT MESSAGES INTO chathistory.txt
def writehistory(text):
    with open('chathistory.txt', 'a') as f:
        f.write(text)
        f.write('\n')
    f.close()


st.title("OSS RAG ChatBot")
st.subheader("intfloat/multilingual-e5-largeとelyza/ELYZA-japanese-Llama-2-7b-fast-instructによるRAGアプリです。最初に参照したいpdfファイルをアップロードしてください")

# Set a default model
# if "hf_model" not in st.session_state:
#     st.session_state["hf_model"] = "MBZUAI/LaMini-Flan-T5-77M"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []



uploaded_file = st.file_uploader('Choose a source pdf')

from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import argparse
import os
if uploaded_file is not None:
    with st.spinner('Wait for indexing...'):
        file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type}
        st.write(file_details)
        with open(os.path.join("resource",uploaded_file.name),"wb") as f: 
            f.write(uploaded_file.getbuffer())
            st.success("Saved File")

        loader = UnstructuredFileLoader(os.path.join("resource",uploaded_file.name))
        documents = loader.load()
        print(f"number of docs: {len(documents)}")
        print("--------------------------------------------------")
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=600,
            chunk_overlap=20,
            separators=["\n\n\n","\n\n","\n"],
        )
        splitted_texts = text_splitter.split_documents(documents)
        print(f"チャンクの総数：{len(splitted_texts)}")
        print(f"チャンクされた文章の確認（1番目にチャンクされたデータ）：\n{splitted_texts[0]}")

        # embed model
        EMBED_MODEL_NAME = "intfloat/multilingual-e5-large"
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

        db = FAISS.from_documents(splitted_texts, embeddings)
        db.save_local("faiss_index/" + uploaded_file.name)
        st.session_state.source = uploaded_file.name
    st.success('indexing completed')

    # Display chat messages from history on app rerun
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    else:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Accept user input
if myprompt := st.chat_input("ご質問をどうぞ"):

    if st.session_state.source == None:
        st.chat_input("先にファイルをアップロードしてください")

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": myprompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(myprompt)
        usertext = f"user: {myprompt}"
        writehistory(usertext)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        apiresponse = requests.get(f'http://127.0.0.1:8000/model?source={st.session_state.source}&question={myprompt}')
        risposta = apiresponse.content.decode("utf-8")
        res  =  risposta[1:-1]
        response = res.split(" ")
        for r in response:
            full_response = full_response + r + " "
            message_placeholder.markdown(full_response + "▌")
            sleep(0.1)
        message_placeholder.markdown(full_response)
        asstext = f"assistant: {full_response}"
        writehistory(asstext)
        st.session_state.messages.append({"role": "assistant", "content": full_response})


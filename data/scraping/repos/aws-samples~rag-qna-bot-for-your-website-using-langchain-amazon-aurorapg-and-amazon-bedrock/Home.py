import streamlit as st
from langchain.vectorstores.pgvector import PGVector
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms.bedrock import Bedrock
from langchain.embeddings import BedrockEmbeddings
from langchain.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain.memory import PostgresChatMessageHistory
from fake_useragent import UserAgent
from bs4 import BeautifulSoup as Soup
import os
import boto3
from botocore.exceptions import ClientError
import tempfile
import time
import random
import hashlib
import json
import secrets

#replace the secret_name and region_name with AWS secret manager where your credentials are stored
def get_secret():
    sm_key_name = "enter-your-secret-key-name"
    region_name = "us-west-2"
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )
    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=sm_key_name
        )
    except ClientError as e:
        print(e)
    secret = get_secret_value_response['SecretString']
    return secret

def generate_session_id():
    t = int(time.time() * 1000)
    r = secrets.randbelow(1000000)
    return hashlib.md5(bytes(str(t) + str(r), 'utf-8'), usedforsecurity=False).hexdigest()

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        chunk_size=512,
        chunk_overlap=103,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# replace the model_id and region_name if you are trying to call a different bedrock model
def get_vectorstore(text_chunks):
    embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",region_name="us-west-2")
    try:
        if text_chunks is None:
            return PGVector(
                connection_string=CONNECTION_STRING,
                embedding_function=embeddings
            )
        return PGVector.from_texts(texts=text_chunks, embedding=embeddings, connection_string=CONNECTION_STRING)
    except Exception as e:
        print(e)
        print(text_chunks)


def get_conversation_chain(vectorstore):
    llm = Bedrock(model_id="anthropic.claude-instant-v1",region_name="us-west-2")
    message_history = PostgresChatMessageHistory(
    connection_string="postgresql://"+secret["username"]+":"+secret["password"]+"@"+secret["host"]+"/genai",
    session_id=generate_session_id())
    memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=message_history, return_source_documents=True, return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    bot_template = "BOT : {0}"
    user_template = "USER : {0}"
    try:
        response = st.session_state.conversation({'question': user_question})
        print(response)
    except ValueError as e:
        st.write(e)
        st.write("Sorry, please ask again in a different way.")
        return

    st.session_state.chat_history = response['chat_history']
    st.write(user_template.replace("{0}", response['question']))
    st.write(bot_template.replace( "{0}", response['answer']))
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{0}", message.content))
        else:
            st.write(bot_template.replace(
                "{0}", message.content))


def main():
    st.title("Build a QnA bot for your website using RAG")
    web_input = st.text_input("Enter an web link and click on 'Process'")
    depth = [1,2,3,4]
    max_depth = st.selectbox("Select the max depth", depth)
    exclude_dir = st.text_input("Enter the subdirectories to exclude(ex: news,weather,learn etc.)")
    exclude_list=[]
    if len(exclude_dir)==0:
        exclude_list=exclude_list
    else:
        exclude = exclude_dir.split(",")
        exclude_list = [web_input + item.strip() for item in exclude]
    if st.button("Process"):
        with st.spinner("Processing"):
            header_template = {}
            header_template["User-Agent"] = UserAgent().random
            loader = RecursiveUrlLoader(url=web_input,headers=header_template,exclude_dirs=exclude_list, max_depth=max_depth, extractor=lambda x: Soup(x, "html.parser").text)
            docs = loader.load()
            for i in docs:
                text_chunks = get_text_chunks(i.page_content)
                #source = [i.metadata]
                vectorstore = get_vectorstore(text_chunks)
            st.session_state.conversation = get_conversation_chain(vectorstore)

    if "conversation" not in st.session_state:
        st.session_state.conversation = get_conversation_chain(get_vectorstore(None))
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

# enter the appropriate DB name
if __name__ == '__main__':  
    secret = json.loads(get_secret())
    CONNECTION_STRING = PGVector.connection_string_from_db_params(                                                  
        driver = "psycopg2",
        user = secret["username"],                                      
        password = secret["password"],                                  
        host = secret["host"],                                            
        port = 5432,                                          
        database = "genai"                                      
    )
    main()
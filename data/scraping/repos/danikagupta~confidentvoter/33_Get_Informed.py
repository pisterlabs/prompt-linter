import streamlit as st
from streamlit_extras.app_logo import add_logo 
from streamlit_extras.switch_page_button import switch_page

import time

import numpy as np

import openai
import pinecone
import streamlit as st
import os

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI



def set_ui_page_get_informed():
    st.image("ConfidentVoter.png")
    #add_logo("http://placekitten.com/120/120")


def load_faiss(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=100,length_function=len, is_separator_regex=False)
    docs = text_splitter.create_documents([text])
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    return db

def augmented_content(inp,info):
    # Create the embedding using OpenAI keys
    # Do similarity search using Pinecone
    # Return the top 5 results
    embedding=openai.Embedding.create(model="text-embedding-ada-002", input=inp)['data'][0]['embedding']
    return info

def show_page_get_informed(ballot_index,ballot_information,ballot_name):
    st.markdown(f"# Get Informed about {ballot_name}")
    SYSTEM_MESSAGE={"role": "system", 
                "content": """
                You are ConfidentVoter - a helpful App that guides the voters about 
                the pros and cons of various issues based on their policy preferences.
                Remember to keep your answers concise and directly addressing the questions asked,
                taking into account the policy preferences that the user has provided.
                """
                }
    ASSISTANT_MESSAGE={"role": "assistant", 
                "content": f"""
                What would you like to know about {ballot_name}?
                Please remember to provide me with your policy preferences so I can provide you with the best possible information.
                """
                }

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append(SYSTEM_MESSAGE)
        st.session_state.messages.append(ASSISTANT_MESSAGE)

    for message in st.session_state.messages:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if prompt := st.chat_input("Help me understand the implications of my vote."):
        #print(f"Prompt: {prompt}")
        retreived_content = augmented_content(prompt,ballot_information)
        #print(f"Retreived content: {retreived_content}")
        prompt_guidance=f"""
    Please guide the user based on the following information from reputable sources:
    {retreived_content}
    The user's question was: {prompt}
        """
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            messageList=[{"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages]
            messageList.append({"role": "user", "content": prompt_guidance})
            
            for response in openai.ChatCompletion.create(
                model="gpt-4",
                messages=messageList, stream=True):
                full_response += response.choices[0].delta.get("content", "")
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
#
#
if __name__ == "__main__":
    set_ui_page_get_informed()
    idx=st.session_state['ballot_index']
    inf=st.session_state['ballot_information']
    mea=st.session_state['measure'];
    ballot_name=st.session_state['ballot_name']
    #st.write(f"Get Informed: IDX={idx} and INF={inf}")
    #st.write(f"Environ={os.environ}")
    show_page_get_informed(idx,inf,ballot_name)
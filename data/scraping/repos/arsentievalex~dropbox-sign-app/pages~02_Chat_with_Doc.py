import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index import download_loader
from llama_index.llms import OpenAI
import openai
import os


st.set_page_config(page_title="ProSign - AI Powered NDA Review & Signing", page_icon="📝", layout="wide", menu_items=None)

page_bg_img = f"""
<style>
  /* Existing CSS for background image */
  [data-testid="stAppViewContainer"] > .main {{
    background-image: url("https://i.postimg.cc/CxqMfWz4/bckg.png");
    background-size: cover;
    background-position: center center;
    background-repeat: no-repeat;
    background-attachment: local;
  }}
  [data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
  }}

  /* New CSS to make specific divs transparent */
  .stChatFloatingInputContainer, .css-90vs21, .e1d2x3se2, .block-container, .css-1y4p8pa, .ea3mdgi4 {{
    background-color: transparent !important;
  }}
</style>
"""

sidebar_bg = f"""
<style>
[data-testid="stSidebar"]{{
    z-index: 1;
}}
</style>
"""


st.markdown(page_bg_img, unsafe_allow_html=True)
st.markdown(sidebar_bg, unsafe_allow_html=True)

openai.api_key = st.secrets["openai_credentials"]["openai_key"]

# initialize to avoid errors
if 'file_name' not in st.session_state.keys():
    st.session_state['file_name'] = None
if 'index' not in st.session_state.keys():
    st.session_state['index'] = None

st.title("ProSign - AI Powered NDA Review & Signing 📝")

if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about the contract!"}
    ]

if st.session_state['index'] is not None:
    chat_engine = st.session_state['index'].as_chat_engine(chat_mode="context", verbose=True)

    # write header
    st.header('Chat with {}'.format(st.session_state['file_name']))

    if prompt := st.chat_input("Your question"):  # Prompt for user input and save to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

    for message in st.session_state.messages:  # Display the prior chat messages
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # If last message is not from assistant, generate a new response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chat_engine.chat(prompt)
                st.write(response.response)
                message = {"role": "assistant", "content": response.response}
                st.session_state.messages.append(message)  # Add response to message history


footer_html = """
    <div class="footer">
    <style>
        .footer {
            position: fixed;
            z-index: 2;
            bottom: 0;
            left: 0;
            right: 0;
            background-color: #283750;
            padding: 10px 20px;
            text-align: center;
        }
        .footer a {
            color: #4a4a4a;
            text-decoration: none;
        }
    </style>
        Made for Dropbox Sign AI Hackathon 2023. Powered by LlamaIndex and OpenAI.
    </div>
"""
st.markdown(footer_html, unsafe_allow_html=True)


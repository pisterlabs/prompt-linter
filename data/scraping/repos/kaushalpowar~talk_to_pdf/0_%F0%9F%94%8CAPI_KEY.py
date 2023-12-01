import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import os
import time
import tempfile
import openai
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, set_global_service_context
from llama_index.llms import OpenAI
from functions import sidebar_stuff1


st.set_page_config(page_title="Talk to PDF", page_icon=":robot_face:", layout="wide")
st.title("Talk to your PDF 🤖 📑️")


st.write("#### Enter your OpenAI api key below :")
api_key = st.text_input("Enter your OpenAI API key (https://platform.openai.com/account/api-keys)", type="password")
st.session_state['api_key'] = api_key

if not api_key :
    st.sidebar.warning("⚠️ Please enter OpenAI API key")
else:
    openai.api_key = api_key

submit = st.button("Submit",use_container_width=True)
if submit:
    st.sidebar.success("✅ API key entered successfully")
    time.sleep(1.5)
    switch_page('upload pdf')
sidebar_stuff1()






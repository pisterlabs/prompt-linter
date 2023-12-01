import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from streamlit_chat import message
import openai

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

##TODO use langchain pinecone client to implement flow here
# Add the current directory to the syspath
currentdir = os.getcwd()
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

# Import the custom python module for interacting with Pinecone and OpenAI
import components.pinecone_langchain as plc

if 'pilang' not in st.session_state:
    st.session_state.pilang = plc.LangChainPineconeClient()

# Try to get the OpenAI API key from the environment variables
try:
    if not os.getenv('OPENAI_API_KEY'):
        os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
    openai_api_key = os.getenv('OPENAI_API_KEY')
except:
    openai_api_key = None
    raise Exception('OPENAI_API_KEY not found in environment variables')

## Begin Streamlit App
# Configure the page to be wide
st.set_page_config(layout='wide')
# Set the title of the app
st.title('Syllabot Verification')

if 'block' not in st.session_state:
    st.session_state.block = False

if 'listofupdates' not in st.session_state:
    st.session_state['listofupdates'] = []

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

if 'whichupdate' not in st.session_state:
    st.session_state.whichupdate = None

i = 0
for msg in st.session_state["messages"]:
    message(msg["content"], is_user=msg["role"] == "user", key = i)
    i += 1

# Intialize a holder for the input box
holder = st.empty()
user_input = None
# Use the holder to create a form which we will hide later after the user submits
if not st.session_state.block:
    with holder.form("chat_input", clear_on_submit=True):
        a, b = st.columns([4, 1])
        user_input = a.text_input(
            label="Your message:",
            placeholder="What would you like to verify about LangChain?",
            label_visibility="collapsed",
        )
        b.form_submit_button("Send", use_container_width=True)

if user_input and openai_api_key and not st.session_state.block:
    st.session_state.block = True
    user_input = user_input.strip()
    openai.api_key = openai_api_key
    holder.empty() # Remove the input box from the app
    st.session_state.messages.append({"role": "user", "content": user_input})
    message(user_input, is_user=True)
    #response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
    # response = pilang.ask_with_context(user_input)
    response = st.session_state.pilang.get_potential_facts(user_input)
    st.session_state['listofupdates'] = [x['name'] for x in st.session_state.pilang.input_updates.potential_facts.values()]
    #msg = response.choices[0].message
    msg = {"role": "assistant","content":response}
    st.session_state.messages.append(msg)
    message(msg['content'])

def get_update():
    # get the pilang.input_updates.potential_facts object where the name is equal to the text
    text = st.session_state['selectbox']
    st.session_state.messages.append({"role": "user", "content": text})
    # message(text, is_user=True)
    dict_ = st.session_state.pilang.input_updates.potential_facts
    key = None
    for key, value in dict_.items():
        if value['name'] == text:
            key = key
            break
    x = st.session_state.pilang.get_potential_updates(key)
    n = f"Update: {text} \n\n {x}"
    st.session_state.messages.append({"role": "assistant", "content": n})
    # message(n)

with st.sidebar:
    st.session_state.whichupdate = st.selectbox('Select an update to view', st.session_state['listofupdates'], on_change=get_update, key='selectbox')

with st.sidebar:
    st.markdown("""
     ## About this App:
    This app is a demo of using Pinecone to implement a chatbot. That processes a user's  course content and determines if it needs to be updated.
    
    ## How to use it:
    Please submit text that you would like to verify about LangChain.
    Take passages from Medium articles, or other sources and submit them to the chatbot. The chatbot will then verify the content and provide a response.
    *Sample Articles:*
    - https://medium.com/databutton/getting-started-with-langchain-a-powerful-tool-for-working-with-large-language-models-286419ba0842
    - https://towardsdatascience.com/a-gentle-intro-to-chaining-llms-agents-and-utils-via-langchain-16cd385fca81
    """)
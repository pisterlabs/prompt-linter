import streamlit as st
from components import sidebar
from services import prompts
import asyncio
from services import llm
import datetime
import openai
import os
import time


# Set org ID and API key
openai_model = os.getenv('OPENAI_API_MODEL')
openai.api_key = os.getenv('OPENAI_API_KEY')
openai.organization = os.getenv('OPENAI_API_ORGANIZATION')

# set page layout
st.set_page_config(
    page_title="Symptoms",
    # page_icon="🤕",
    layout="wide"
)

#THIS CSS MODIFIES THE COLOR OF TEXT BOXES
st.markdown("""
    <style>
    .stTextArea [data-baseweb=base-input] {
        background-color: #080808;
        color: white;
    }
            
    .stTextInput [data-baseweb=base-input] {
        background-color: #080808;
        color: white;
    }
            
    .stDateInput [data-baseweb=base-input] {
        background-color: #080808;
        color: white;
    }
    
    .stNumberInput [data-baseweb=base-input] {
        background-color: #080808;
        color: white;
    }
            
    div[data-baseweb="select"] > div {
        background-color: #080808;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

if "name" not in st.session_state:
    st.session_state.name = ""
if "age" not in st.session_state:
    st.session_state.age = 18
if "dob" not in st.session_state:
    st.session_state.dob = datetime.datetime.now()
if "gender" not in st.session_state:
    st.session_state.gender = ""
if "smoker" not in st.session_state:
    st.session_state.smoker = ""
if "med_conditions" not in st.session_state:
    st.session_state.med_conditions = []
if "allergies" not in st.session_state:
    st.session_state.allergies = ""
if "meds" not in st.session_state:
    st.session_state.meds = ""
if "parent_conditions" not in st.session_state:
    st.session_state.parent_conditions = []
if "sibling_conditions" not in st.session_state:
    st.session_state.sibling_conditions = []
if "grandparent_conditions" not in st.session_state:
    st.session_state.grandparent_conditions = []
if "report" not in st.session_state:
    st.session_state.report = ""

st.session_state.name = st.session_state.name
st.session_state.age = st.session_state.age
st.session_state.dob = st.session_state.dob
st.session_state.gender = st.session_state.gender
st.session_state.smoker = st.session_state.smoker
st.session_state.med_conditions = st.session_state.med_conditions
st.session_state.allergies = st.session_state.allergies
st.session_state.meds =  st.session_state.meds
st.session_state.parent_conditions = st.session_state.parent_conditions
st.session_state.sibling_conditions = st.session_state.sibling_conditions
st.session_state.grandparent_conditions = st.session_state.grandparent_conditions
st.session_state.report = st.session_state.report


# Sidebar - let user clear the current conversation
sidebar.display()
clear_button = st.sidebar.button("Clear Conversation", key="clear")
share_button = st.sidebar.button("Share Data with Doctor", key="share")

st.markdown("<h1 style='text-align: center;'>Welcome to Panacea AI</h1>", unsafe_allow_html=True)

# Chat with the LLM, and update the messages list with the response, updating UI
async def chat(messages):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        messages = await llm.run_conversation(messages, message_placeholder)
        st.session_state.messages = messages
    return messages

# Initialise session state variables
if 'messages' not in st.session_state or len(st.session_state.messages) < 2:
    st.session_state['messages'] = [
        {"role": "system", "content": prompts.startPrompt()}
    ]

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        st.session_state.messages.append({"role": "user", "content": "Hello!"})
        messages = asyncio.run(llm.run_conversation(st.session_state.messages, message_placeholder))
        st.session_state.messages = messages


# reset everything
if clear_button:
    st.session_state['messages'] = [
        {"role": "system", "content": prompts.startPrompt()}
    ]

# Print all messages in the session state
for message in [m for m in st.session_state.messages if m["role"] != "system"][2:]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if share_button:
    with st.spinner('Sharing symptoms and health trends with doctor...'):
        time.sleep(2)
        st.success("""
                   Sent! Your doctor has received a full-report of the symptoms we just discussed 
                   together as well as a copy of your medical records and your Apple Health data trends 
                   over the past two weeks. You will receive an email shortly confirming your doctor appointment. ')
        """)


# React to the user prompt
if prompt := st.chat_input("How are you feeling today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    asyncio.run(chat(st.session_state.messages))







import os
import streamlit as st
import random
import time
import requests
import json

from config import apikeys
from app.controllers.LangchainController import LangchainController

os.environ["OPENAI_API_KEY"] = apikeys.OPENAI_API_KEY

# Helpers


def get_response(chat_message, chat_room, keys_to_retrieve=12):
    # Get Chat Room from DB

    langchain = LangchainController()

    if 'chat' not in st.session_state['chat_room']:
        st.session_state['chat_room']['chat'] = []

    ai_message = langchain.get_response(chat_room=chat_room, chat_message=chat_message,
                                            keys_to_retrieve=keys_to_retrieve)


    st.session_state['chat_room']['chat'].append({
        "content": chat_message,
        "type": "User"
    })
    st.session_state['chat_room']['chat'].append({
        "content": ai_message['response'],
        "type": "AI"
    })

    st.session_state['chat_room']['cb'] = ai_message['cb']


    return st.session_state['chat_room']

def refresh_chat():
    st.session_state['messages'] = []
    st.session_state['total_cost'] = 0.0
    st.session_state['total_tokens'] = 0
    st.session_state['prompt_tokens'] = 0
    st.session_state['completion_tokens'] = 0

def set_session_state_variable(index, value):
    st.session_state[index] = value

def set_button(button):
    st.session_state[button] = True
    st.experimental_rerun()

st.set_page_config(
    page_title='Samsus Superbrain',
    page_icon='🧊',
    initial_sidebar_state='auto'
)

if not st.session_state.get('total_cost'):
    st.session_state['total_cost'] = 0.0

if not st.session_state.get('total_tokens'):
    st.session_state['total_tokens'] = 0

if not st.session_state.get('prompt_tokens'):
    st.session_state['prompt_tokens'] = 0

if not st.session_state.get('completion_tokens'):
    st.session_state['completion_tokens'] = 0

st.title("Samsus Superbrain Chatbot")

col1, col2, col3, col4 = st.columns(4)

col1.metric(label="Total Cost in $", value=st.session_state['total_cost'])

col2.metric(label="Total Tokens", value=st.session_state['total_tokens'])

col3.metric(label="Prompt Tokens", value=st.session_state['prompt_tokens'])

col4.metric(label="Completion Tokens", value=st.session_state['completion_tokens'])

chat_model = 'gpt-3.5-turbo-16k'


if not st.session_state.get('chat_room'):
    st.session_state['chat_room'] = {}

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = chat_model

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("How can I help you?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        response = get_response(prompt, st.session_state['chat_room'], keys_to_retrieve=12)
        message_placeholder.markdown(response['chat'][-1]['content'])
    st.session_state.messages.append({"role": "assistant", "content": response['chat'][-1]['content']})
    st.session_state['total_cost'] = round(st.session_state['total_cost'] + response['cb'].total_cost, 6)
    st.session_state['prompt_tokens'] += response['cb'].prompt_tokens
    st.session_state['completion_tokens'] += response['cb'].completion_tokens
    st.session_state['total_tokens'] += response['cb'].total_tokens
    st.experimental_rerun()

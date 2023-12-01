# https://www.youtube.com/watch?v=KBo7mZHlink&t=9s

import os
import openai
import streamlit as st
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from common import placeholder_msg, SYSTEM_MESSAGE_CHI, set_gpt3_env, set_gpt4_env
placeholder_msg= placeholder_msg

st.set_page_config(page_title="Chinese Chatbot", page_icon="🇨🇳")
st.markdown("# 🇨🇳 Custom Chinese Chatbot")

with st.expander("🤖 See System Message"):
    st.write(SYSTEM_MESSAGE_CHI)

with st.sidebar:

    option_ai = st.radio(
        "Select AI",
        key="ai",
        options=["GPT3", "GPT4"],
    )

    clear_button = st.sidebar.button("Clear Conversation", key="clear")

# Set up the environment for GPT3 or GPT4
if option_ai == "GPT3":
    set_gpt3_env()
else:
    set_gpt4_env()


SYSTEM_MESSAGE = SYSTEM_MESSAGE_CHI

if "messages_chi" not in st.session_state:
    st.session_state.messages_chi = []

if clear_button:
    st.session_state.messages_chi = []


for message in st.session_state.messages_chi:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input(placeholder_msg):
    st.session_state.messages_chi.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        response = openai.ChatCompletion.create(
            engine=st.session_state["openai_model"],
            messages=[{"role": "system", "content": SYSTEM_MESSAGE}] +
                     [
                         {"role": m["role"], "content": m["content"]}
                         for m in st.session_state.messages_chi
                     ],
            # stream=True
        )
        full_response += response.choices[0].message.content
        message_placeholder.markdown(full_response)
    st.session_state.messages_chi.append({"role": "assistant", "content": full_response})

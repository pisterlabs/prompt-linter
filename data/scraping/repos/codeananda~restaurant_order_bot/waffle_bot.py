"""A chatbot that helps the user order food from a restaurant."""

from copy import deepcopy

import streamlit as st
from openai import OpenAI

from utils import generate_response, initial_state

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Top matter
st.set_page_config(page_title="Waffle House Order Bot", page_icon=":waffle:")
st.title("🧇 Waffle Restaurant Order Bot")

intro = """Welcome to the Waffle House, the place where all your waffle dreams come true.

Start chatting with WaffleBot below to find out what you can order, how much it costs, and how to pay."""
st.markdown(intro)

if "messages" not in st.session_state:
    st.session_state["messages"] = deepcopy(initial_state)

# Let user clear the current conversation
clear_button = st.button("Clear Conversation", key="clear")
if clear_button:
    st.session_state["messages"] = deepcopy(initial_state)

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    if message["role"] == "system":
        continue
    elif message["role"] == "assistant":
        with st.chat_message(message["role"], avatar="🧇"):
            st.markdown(message["content"])
    else:
        with st.chat_message(message["role"], avatar="👤"):
            st.markdown(message["content"])

if prompt := st.chat_input("What would you like to order?"):
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    output = generate_response(prompt)
    with st.chat_message("assistant", avatar="🧇"):
        st.markdown(output)

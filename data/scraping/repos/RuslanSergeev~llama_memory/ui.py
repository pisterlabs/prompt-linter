# streamlit ui for llama_memory.
# uses chat-api only. memory accessed via agent calls.

import os
import openai
import streamlit as st

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading memory..."):
        from llama_memory import Llama_memory
        memory = Llama_memory('storage')
        return memory

@st.cache_resource(show_spinner=False)
def load_keys():
    with st.spinner(text="Loading keys..."):
        return os.environ.get("OPENAI_API_KEY")

memory = load_data()
openai.api_key = load_keys()

st.title("Llama memory")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Simulate bot typing
    full_response = ""
    with st.chat_message("assistant"):
        placeholder = st.empty()
        with st.spinner("Thinking..."):
            rsp = memory.chat_stream(prompt)
            for chunk in rsp.response_gen:
                full_response += chunk
                placeholder.markdown(full_response)
        placeholder.empty()
        st.markdown(full_response)
        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)

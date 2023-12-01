# Implement demo code by Streamlit: https://github.com/streamlit/llm-examples/tree/main

import streamlit as st
import pandas as pd
import openai

# Load settings
openai_api_key = st.secrets["openai"]["key"]

# Display app basic information
st.title("💬 Basic ChatGPT Example")
st.caption("🚀 Interact with the OpenAI ChatGPT API")

# Initialize the web session
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# Execute
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    openai.api_key = openai_api_key
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
    msg = response.choices[0].message
    st.session_state.messages.append(msg)
    st.chat_message("assistant").write(msg.content)


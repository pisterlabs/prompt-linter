import os
import openai
import streamlit as st

from dotenv import load_dotenv

_ = load_dotenv()

st.set_page_config(page_title="Chatbot Basic", page_icon="💬")

with st.sidebar:
    openai_api_type = st.text_input(
        "OpenAI API Type", value=os.environ["OPENAI_API_TYPE"], key="chatbot_api_type"
    )
    openai_api_base = st.text_input(
        "OpenAI API Base", value=os.environ["OPENAI_API_BASE"], key="chatbot_api_base"
    )
    openai_api_version = st.text_input(
        "OpenAI API Version",
        value=os.environ["OPENAI_API_VERSION"],
        key="chatbot_api_version",
    )
    openai_api_key = st.text_input(
        "OpenAI API Key",
        value=os.environ["OPENAI_API_KEY"],
        key="chatbot_api_key",
        type="password",
    )
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

st.title("💬 Chatbot Basic")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    openai.api_key = openai_api_key
    openai.api_base = openai_api_base
    openai.api_type = openai_api_type
    openai.api_version = openai_api_version

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    response = openai.ChatCompletion.create(
        engine=os.environ["DEPLOYMENT_NAME"], messages=st.session_state.messages
    )
    msg = response.choices[0].message
    st.session_state.messages.append(msg)
    st.chat_message("assistant").write(msg.content)

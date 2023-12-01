# pylint: disable=invalid-name
# pylint: disable=non-ascii-file-name
""" ChatGPT example """

import os

import openai
import streamlit as st


def clear_chat():
    """Clear chat"""
    st.session_state.messages = []


st.set_page_config(page_title="Ask ChatGPT", page_icon="🙊")
st.title("🙊 Ask ChatGPT")

if st.button("New chat"):
    clear_chat()

st.markdown(
    "This example is a simple chatbot that uses the "
    "[OpenAI API](https://platform.openai.com/docs/models/overview) "
    "and GPT models to generate responses to your questions."
)

gpt_model = st.selectbox("Model", ["gpt-3.5-turbo", "gpt-4"])

if os.environ.get("OPENAI_API_KEY") is not None:
    openai.api_key = os.getenv("OPENAI_API_KEY")
else:
    st.error("OPENAI_API_KEY environment variable not set")
    st.stop()

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = gpt_model

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Send a message"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in openai.ChatCompletion.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        ):
            full_response += response.choices[0].delta.get("content", "")
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )

import openai
import streamlit as st
from utils import openai_call
import os

st.title("24/7 Virtual Assistant")

GPT_MODEL = "gpt-3.5-turbo-16k"
openai.api_key = st.secrets["OPENAI_API_KEY"]


if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "transcripts" not in st.session_state:
    st.session_state.transcripts = [
        files for files in os.listdir("transcripts") if files.endswith(".txt")
    ]

transcript_file = st.selectbox(
    "Select a transcript file", options=st.session_state.transcripts
)

with open("transcripts/" + transcript_file, "r") as f:
    lecture = f.read()


for message in st.session_state.chat_messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


st.session_state.chat_messages.extend(
    [
        {
            "role": "system",
            "content": f"Your task is to answer questions on this lecture: {lecture}.",
        },
    ]
)


if prompt := st.chat_input("What is up?"):
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        content = openai_call(
            st.session_state.chat_messages, message_placeholder, model=GPT_MODEL
        )

    st.session_state.chat_messages.append({"role": "assistant", "content": content})

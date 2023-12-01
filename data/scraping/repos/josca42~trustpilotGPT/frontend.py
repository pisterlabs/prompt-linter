import streamlit as st
import openai
from assistant.chat import chat
from assistant.config import config

st.title("ChatGPT-like clone")

openai.api_key = config["OPENAI_API_KEY"]

with st.sidebar:
    new_chat = st.button(
        ":heavy_plus_sign: New chat", key="new_chat", use_container_width=True
    )
    st.sidebar.button(":exclamation: Stop generating", use_container_width=True)

    st.caption("Today")
    st.button("Enter some text", key="1", use_container_width=True)

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(message["content"])
    elif message["role"] == "assistant":
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(message["content"])
    elif message["role"] == "plot":
        st.plotly_chart(message["content"], use_container_width=True)
    else:
        pass

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    msgs_chat_input = [m for m in st.session_state.messages if m["role"] != "plot"][-3:]
    msgs_chat_output = chat(messages=msgs_chat_input, st=st)
    st.session_state.messages.extend(msgs_chat_output)

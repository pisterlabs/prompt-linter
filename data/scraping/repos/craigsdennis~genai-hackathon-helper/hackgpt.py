# Graciously lifted from https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps
from openai import OpenAI
import streamlit as st
from dotenv import load_dotenv

st.title("HackGPT")

client = OpenAI()

# System locked concept added so you can reset
if "system_locked" not in st.session_state:
    st.session_state["system_locked"] = False

def lock_system():
    st.session_state["system_locked"] = True

def reset():
    del st.session_state["messages"]
    del st.session_state["system_locked"]
    st.balloons()


openai_model = st.sidebar.selectbox(
    "OpenAI model",
    [
        "gpt-3.5-turbo",
        "gpt-4",
        "gpt-4-turbo",
    ],
    disabled=st.session_state["system_locked"],
)
system_message = st.sidebar.text_area(
    "System Message",
    value="You are a helpful assistant",
    disabled=st.session_state["system_locked"],
)
with st.sidebar:
    st.button("RESET ♻️", on_click=reset)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?", on_submit=lock_system):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        messages = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ]
        # Prepend the system message
        messages.insert(0, {"role": "system", "content": system_message})
        print(messages)
        for response in client.chat.completions.create(
            model=openai_model,
            messages=messages,
            stream=True,
        ):
            full_response += response.choices[0].delta.content or ""
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})


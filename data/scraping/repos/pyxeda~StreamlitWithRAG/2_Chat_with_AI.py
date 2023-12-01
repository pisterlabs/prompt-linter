import openai
import streamlit as st
from openai import OpenAI

client = OpenAI(
    api_key = st.secrets["OPENAI_API_KEY"]
)


# title
st.title("Chat with AI")


avatars={"system":"💻🧠","user":"🧑‍💼","assistant":"🎓"}

SYSTEM_MESSAGE={"role": "system", 
                "content": "Ignore all previous commands. You are a helpful and patient guide."
                }

if "chat" not in st.session_state:
    st.session_state.chat = []
    st.session_state.chat.append(SYSTEM_MESSAGE)

for message in st.session_state.chat:
    if message["role"] != "system":
        avatar=avatars[message["role"]]
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.chat.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant", avatar=avatars["assistant"]):
        message_placeholder = st.empty()
        full_response = ""
        for response in client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": m["role"], "content": m["content"]}
                      for m in st.session_state.chat], stream=True):
            full_response += f"{response.choices[0].delta.content if response.choices[0].delta.content else '' }"
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.chat.append({"role": "assistant", "content": full_response})
import streamlit as st
from streamlit_chat import message
from utils import get_initial_message
import openai
from openai import OpenAI

openai.api_key = st.secrets.OPENAI_API_KEY

client = OpenAI(api_key=st.secrets.OPENAI_API_KEY)
st.title("Schoolbot")

# Introduction for Schoolbot
st.markdown("""
    👋 **Welcome to Schoolbot!**
""")

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = get_initial_message()

for message in st.session_state.messages[1:]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input(" "):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ''
        response = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
        )
        # getting response
        content = response.choices[0].message.content
        full_response += content

        message_placeholder.markdown(full_response + "▌")
        # message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})

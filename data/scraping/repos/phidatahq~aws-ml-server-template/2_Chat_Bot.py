from os import getenv
from typing import Optional, List, Dict

import openai
import streamlit as st
from streamlit_chat import message

#
# -*- Get OpenAI API key
#
# Get OpenAI API key from environment variable
openai_api_key: Optional[str] = getenv("OPENAI_API_KEY")
# If not found, get it from user input
if openai_api_key is None:
    text = "Enter OpenAI API key"
    st.text_input("Enter Openai API key", value=text, key="api_key")
    openai_api_key = text


def generate_response(messages: List[Dict[str, str]]) -> str:
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.1,
        # stream=True,
        max_tokens=2048,
    )
    # st.write(completion)
    response = completion.choices[0].message
    return response


def chatbot_demo():
    # Create session variable to store the chat
    if "all_messages" not in st.session_state:
        st.session_state["all_messages"] = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]

    user_message = st.text_input("Message:", key="input")
    if user_message:
        new_message = {"role": "user", "content": user_message}
        st.session_state["all_messages"].append(new_message)

        # Generate response
        output = generate_response(st.session_state["all_messages"])
        # Store the output
        st.session_state["all_messages"].append(output)

    if st.session_state["all_messages"]:
        for msg in st.session_state["all_messages"]:
            if msg["role"] == "user":
                message(msg["content"], is_user=True)
            elif msg["role"] == "assistant":
                message(msg["content"])


st.markdown("# Chatting with GPT-3.5 Turbo")
st.write("This is a chatbot built using OpenAI. Send it a message and it will respond.")

st.sidebar.header("Chatbot Demo")
chatbot_demo()

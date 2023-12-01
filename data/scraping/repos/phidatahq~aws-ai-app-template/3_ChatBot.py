from os import getenv, environ
from typing import Optional, List, Dict

import openai
import streamlit as st
from streamlit_chat import message


#
# -*- Sidebar component to get OpenAI API key
#
def get_openai_key() -> Optional[str]:
    # Get OpenAI API key from environment variable
    OPENAI_API_KEY: Optional[str] = getenv("OPENAI_API_KEY")
    # If not found, get it from user input
    if OPENAI_API_KEY is None or OPENAI_API_KEY == "" or OPENAI_API_KEY == "sk-***":
        api_key = st.sidebar.text_input("OpenAI API key", value="sk-***", key="api_key")
        if api_key != "sk-***":
            OPENAI_API_KEY = api_key
            st.session_state["OPENAI_API_KEY"] = OPENAI_API_KEY
            environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    # Store it in session state and environment variable
    if OPENAI_API_KEY is not None:
        st.session_state["OPENAI_API_KEY"] = OPENAI_API_KEY
        environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    return OPENAI_API_KEY


#
# -*- Sidebar component to show reload button
#
def show_reload():
    st.sidebar.markdown("---")
    if st.sidebar.button("Reload Session"):
        st.session_state.clear()
        st.experimental_rerun()


#
# -*- ChatBot Sidebar
#
def chatbot_sidebar():
    st.sidebar.markdown("# Chatbot")

    # Get OpenAI API key
    openai_key = get_openai_key()
    if openai_key is None:
        st.write("🔑  OpenAI API key not set")

    # Show reload button
    show_reload()


def generate_response(messages: List[Dict[str, str]]) -> str:
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0,
        # stream=True,
        max_tokens=2048,
    )
    # st.write(completion)
    response = completion.choices[0].message
    return response


#
# -*- ChatBot Main UI
#
def chatbot_main():
    user_message = st.text_input(
        "Send a message:",
        placeholder="Write a python function to add two numbers",
        key="user_message",
    )
    if user_message:
        # Create a session variable to store messages
        if "all_messages" not in st.session_state:
            st.session_state["all_messages"] = [
                {
                    "role": "system",
                    "content": """You are a helpful assistant that answers questions about programming in python""",  # noqa: E501
                },
            ]

        new_message = {"role": "user", "content": user_message}
        st.session_state["all_messages"].append(new_message)

        # Generate response
        output = generate_response(st.session_state["all_messages"])
        # Store the output
        st.session_state["all_messages"].append(output)

    if "all_messages" in st.session_state:
        for msg in st.session_state["all_messages"]:
            if msg["role"] == "user":
                message(msg["content"], is_user=True)
            elif msg["role"] == "assistant":
                message(msg["content"])


#
# -*- ChatBot UI
#
st.markdown("## ChatBot")
st.write(
    """This is a chatbot built using OpenAI, customize it to your needs.\n
    Send it a message and it will respond.
    """
)

chatbot_sidebar()
chatbot_main()

import os
import streamlit as st
from dotenv import load_dotenv
import openai

load_dotenv()


def generate_assistant_response():
    response = openai.ChatCompletion.create(
        model=st.session_state["openai_model"], messages=st.session_state.messages
    )
    return response


# .choices[0].message["content"]


def main():
    st.title("ChatGPT-like clone")
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"

    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("What is up?")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        assistant_response = generate_assistant_response()
        with st.chat_message("assistant"):
            st.markdown(assistant_response)

        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_response}
        )
        st.session_state.messages


if __name__ == "__main__":
    main()

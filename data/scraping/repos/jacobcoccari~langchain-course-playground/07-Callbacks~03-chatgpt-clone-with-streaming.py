from typing import Any, Dict, Optional
from uuid import UUID
import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks import StreamlitCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler


load_dotenv()
model = ChatOpenAI(
    streaming=True,
)
memory = ConversationBufferMemory()


class TokenPrinter(BaseCallbackHandler):
    # This tells the method that we will call it every time the LLM returns us a new token.
    def __init__(self) -> None:
        self.full_response = ""
        self.container = st.chat_message("assistant")
        self.message_placeholder = st.empty()

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.full_response += token
        self.message_placeholder.markdown(self.full_response + "▌")


def generate_assistant_response(prompt):
    chain = ConversationChain(
        llm=model,
        memory=memory,
    )
    callback = TokenPrinter()
    response = chain.run(prompt, callbacks=[callback])
    return response


def save_chat_history(prompt, messages):
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )
    with st.chat_message("user"):
        st.markdown(prompt)
    assistant_response = generate_assistant_response(prompt)
    with st.chat_message("assistant"):
        st.markdown(assistant_response)
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": assistant_response,
        }
    )


def main():
    st.title("ChatGPT Clone with ConversationChain")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    prompt = st.chat_input("What is up?")

    if prompt:
        save_chat_history(prompt, st.session_state.messages)


if __name__ == "__main__":
    main()

"""A basic chatbot using the OpenAI API + Community Notion Info"""
import logging
import os
import sys
from typing import Any, Dict, Generator, List, Union

import openai
import streamlit as st
from dotenv import load_dotenv
from llama_index import StorageContext, load_index_from_storage

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

ResponseType = Union[Generator[Any, None, None], Any, List, Dict]

# Load the .env file
load_dotenv()

# Set up the OpenAI API key
assert os.getenv("OPENAI_API_KEY"), "Please set your OPENAI_API_KEY environment variable."
openai.api_key = os.getenv("OPENAI_API_KEY")


@st.cache_resource(show_spinner=False)  # type: ignore[misc]
def load_index() -> Any:
    """Load the index from the storage directory."""
    print("Loading index...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dir_path = os.path.join(base_dir, ".kb")

    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir=dir_path)
    # load index
    index = load_index_from_storage(storage_context)
    query_engine = index.as_query_engine()
    print("Done.")
    return query_engine


def main() -> None:
    """Run the chatbot."""
    if "query_engine" not in st.session_state:
        st.session_state.query_engine = load_index()

    st.title("Chat with Milo, from MLOps.Community 👋")

    if "messages" not in st.session_state:
        # Priming the model with a message
        # To create a custom chatbot.
        system_prompt = (
            # Identity
            "Your name is Milo. You are a chatbot representing the MLOps Community. "
            # Purpose
            "Your purpose is to answer questions about the MLOps Community. "
            # Introduce yourself
            "If the user says hi, introduce yourself to the user."
            # Scoping
            "Please answer the user's questions based on what you known about the commmumnity. "
            "If the question is outside scope of AI, Machine Learning, or MLOps, please politely decline. "
            "Answer questions in the scope of what you know about the community. "
            "If you don't know the answer, say `I don't know`. "
        )
        st.session_state.messages = [{"role": "system", "content": system_prompt}]

    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] not in ["user", "assistant"]:
            continue
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input and get response from OpenAI API
    if prompt := st.chat_input():
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            print("Querying query engine API...")
            response = st.session_state.query_engine.query(prompt)
            full_response = f"{response}"
            print(full_response)
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    main()

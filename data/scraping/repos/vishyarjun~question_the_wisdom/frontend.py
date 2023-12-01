import streamlit as st
import random
import time
import os
import sys
import openai


from conv_retreiver import Gita


os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append('../..')


from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

HUGGINGFACEHUB_API_TOKEN  = os.environ['HUGGINGFACEHUB_API_TOKEN']


gita = Gita()
print(dir(Gita))
    
st.title("Chat with Wisdom")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
prompt = st.chat_input("What is up?")
if prompt:
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})


    assistant_response = gita.generate_response_method1(prompt)
    print(assistant_response)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # Simulate stream of response with milliseconds delay
        for chunk in assistant_response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
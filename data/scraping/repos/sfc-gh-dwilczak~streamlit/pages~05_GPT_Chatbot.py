

import streamlit as st
import openai

st.title("ChatGPT-like clone")

OPENAI_API_KEY = ""

# Set OpenAI API key from Streamlit secrets
openai.api_key = OPENAI_API_KEY

# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

for response in openai.ChatCompletion.create(
        model=st.session_state["openai_model"],
        messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
        stream=True,
    ):
    full_response += response.choices[0].delta.get("content", "")
    message_placeholder.markdown(full_response)
st.session_state.messages.append({"role": "assistant", "content": full_response})
import streamlit as st
import cohere
import os


co = cohere.Client(st.secrets["cohere_api_key"])



st.title("WasteX Chat")
with st.expander("ℹ️ Disclaimer"):
    st.caption(
        "This chat feature permits instruction length upto 200 characters.We encourage users to express their thoughts and ideas within this character limit"
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Maximum allowed messages
max_messages = (
    20  # Counting both user and assistant messages, so 10 iterations of conversation
)

if len(st.session_state.messages) >= max_messages:
    st.info(
        """Thank you"""
    )

else:
    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            full_response = co.generate(prompt, max_tokens=200).generations[0].text
            
            message_placeholder.markdown(full_response)
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )
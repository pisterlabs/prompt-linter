import openai
import streamlit as st
from openai.error import AuthenticationError
from config.config import OPENAI_PARAMS
from components.sidebar import sidebar

# Page Configuration
st.set_page_config(page_title="Eliza Chatbot", page_icon="img/eliza_logo.jpg", layout="wide")
logo_image = "img/eliza_logo.jpg"  
st.image(logo_image,width=150)  
project_name = "ElizaBot: A Conversational Psychotherapy Chatbot."  
st.title(project_name)  
st.markdown(
    "This mini-app replicates [Eliza's](https://dl.acm.org/doi/10.1145/365153.365168) chatbot behaviour using OpenAI's [GPTs](https://beta.openai.com/docs/models/overview)"
)

#Add sidebar
sidebar()

# Warning to make the user input their API key
openai_api_key = st.session_state.get("OPENAI_API_KEY")
if not openai_api_key:
    st.warning(
        "Enter your OpenAI API key in the sidebar. If you don't have any, select 'No' to the sidebar's question.", icon="⚠️"
    )
openai.api_key = openai_api_key

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Share your feelings"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        try:
            for response in openai.ChatCompletion.create(
                model=OPENAI_PARAMS['openai_model'],
                messages= [{"role": "system", "content": OPENAI_PARAMS['context']}] 
                + [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                stream=True,
                temperature = OPENAI_PARAMS['temperature'],
                max_tokens= OPENAI_PARAMS['max_tokens'],
                top_p= OPENAI_PARAMS['top_p'],
                frequency_penalty = OPENAI_PARAMS['frequency_penalty'],
                presence_penalty= OPENAI_PARAMS['presence_penalty'],

            ):
                full_response += response.choices[0].delta.get("content", "")
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except AuthenticationError as auth_error:
            # Handle the AuthenticationError 
            st.warning(
                "Looks like the OpenAI API key is not valid."
            )
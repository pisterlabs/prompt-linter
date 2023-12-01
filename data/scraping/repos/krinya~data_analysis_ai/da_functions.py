import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import json
import re
import openai
#from openai.embeddings_utils import get_embedding, cosine_similarity


def generate_answer_using_context(query, context, conversation):

    try:
        openai.api_key = st.secrets['open_ai']['api_key']
        api_key_custom_input = st.secrets['open_ai']['api_key']
    except:
        openai.api_key  = st.session_state['openai_api_key']
        api_key_custom_input = st.session_state['openai_api_key']

    if api_key_custom_input == "":
        st.error("Please enter your OpenAI API key on the front page.")
        return "No API key entered.", "No API key entered."

    if (conversation is None) or (conversation == ""):
        st.session_state.previus_conversations = [
            {"role": "system", "content": "You are a helpful assistant that writes code and queries for the users."},
            {"role": "assistant", "content": context}
        ]

    st.session_state.previus_conversations.append(
        {"role": "user", "content": query}
    )

    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=st.session_state.previus_conversations,
        temperature=0.7,
        max_tokens=1000,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    answer_message = response['choices'][0]['message']
    st.sidebar.write(response['usage'])
    st.session_state.previus_conversations.append(answer_message)

    return answer_message['content'], st.session_state.previus_conversations

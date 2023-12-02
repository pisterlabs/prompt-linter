from typing import Callable

import dotenv
import streamlit as st
from langchain.chains import ConversationChain

from {{cookiecutter.package_name}}.ai.chains import get_basic_conversation_chain
from {{cookiecutter.package_name}}.streamlit.llm_blocks import llm_chatbot_st_block

dotenv.load_dotenv(".env", override=True)


@st.cache_resource
def get_chatbot_resource() -> ConversationChain:
    return get_basic_conversation_chain(model_name='gpt-3.5-turbo')


st.set_page_config(page_title="Chatbot Demo", page_icon="🤖")

st.info(
    '''This page demos a simple Chatbot. 
Chat with it and watch it keep track of the conversation.''',
    icon="ℹ️"
)
llm_chatbot_st_block("ChatBot", get_chatbot_resource())

import time

import streamlit as st
from friend_replica.format_chat import ChatConfig, format_chat_history, split_chat_data
from friend_replica.recollection import LanguageModelwithRecollection
from friend_replica.semantic_search import *
from langchain.llms import GPT4All
from model_paths import path_en
from models.model_cn import ChatGLM

### Side Bar Module ###
with st.sidebar:
    "[Get a Comma API key](https://github.com/roxie-zhang/friend_replica)"
    "[View the source code](https://github.com/roxie-zhang/friend_replica)"

### Header Module ###
st.title("Comma Friend Replica - Recollection")
st.caption("🚀 Recollection helps you to summarize "
           "| *FDU Comma Team Ver-1.1*")
# st.markdown('---')

### Config Model ###
st.subheader('Chat History')


# Load Memory Recollection Model
if st.session_state.language == 'chinese':
    model = ChatGLM()
else: 
    model = GPT4All(model=path_en)
m = LanguageModelwithRecollection(model, st.session_state.chat_with_friend, debug=True)

# %%
# Memory Archive Generation
# m.memory_archive(chat_blocks)

# For one Chat Block
# st.write('\n'.join(format_chat_history(st.session_state.chat_blocks[1],
#                                        st.session_state.chat_with_friend.chat_config,
#                                        for_read=True,
#                                        time=True)))

st.text('\n'.join(format_chat_history(st.session_state.chat_blocks[0],
                                       st.session_state.chat_with_friend.chat_config,
                                       for_read=True,
                                       time=True)))
st.subheader('Chat Summarization')

def summarize_memory():
    return m.summarize_memory(st.session_state.chat_blocks[0])

st.text(summarize_memory())
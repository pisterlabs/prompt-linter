import streamlit as st
from streamlit.logger import get_logger
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.chains.conversation.memory import ConversationBufferMemory
import os

from RAG import create_qa_chain
from utils import add_sidebar


LOGGER = get_logger(__name__)

st.set_page_config(page_title="MadKudu: Support Rubook Chat", page_icon=":robot_face:")
st.title("MadKudu: Chat with our Notion Support Runbooks")

add_sidebar(st)

qa_chain = create_qa_chain()

# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory()
qa_chain.memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
    msgs.clear()
    msgs.add_ai_message("How can I help you?")

avatars = {"human": "user", "ai": "assistant"}
for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

if user_query := st.chat_input(placeholder="Ask me anything!"):
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):        
        response = qa_chain.run(user_query, callbacks=[msgs])


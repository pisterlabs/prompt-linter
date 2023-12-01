import streamlit as st
import time
import os
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from origin.components.sidebar import sidebar
from utils import config

from dotenv import load_dotenv

load_dotenv()


# 设置页面标题
st.set_page_config(page_title="Chat-GPT", page_icon="🤖", layout="wide")
st.title("Chat-GPT")

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', config('OPENAI_API_KEY'))
chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=.7, max_tokens=2000, model_name='gpt-3.5-turbo', streaming=True, callbacks=[StreamingStdOutCallbackHandler()])

sidebar()

# 模拟ChatGPT的响应
def chatgpt_response(user_message):
    messages = [
        HumanMessage(content=user_message)

    ]
    return chat(messages).content

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if user_message := st.chat_input("What’s the haps?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        print("guy:" + user_message)
        st.markdown(user_message)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_message})
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        bot_response = ""
        for response in chatgpt_response(user_message):
            # 使用chatgpt_response函数处理用户消息
            bot_response += response
            # Add a blinking cursor to simulate typing
            time.sleep(0.01)
            message_placeholder.markdown(bot_response + "▌")
        message_placeholder.markdown(bot_response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": bot_response})

import streamlit as st

from streamlit_chat import message
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI

from modules.state import read_url_param_values
from Home import APP_TITLE, APP_ICON


st.set_page_config(
    page_title=f"{APP_TITLE} - OpenAI Chat",
    page_icon=APP_ICON
)


def setup_chat_session():
    # Storing the chat
    if "chat" not in st.session_state:
        st.session_state["chat"] = []


def generate_chat_response(prompt):
    config = read_url_param_values()
    memory = memory_buffer()

    llm = ChatOpenAI(
        temperature=config["temperature"],
        openai_api_key=config["api_key"],
        max_tokens=config["max_tokens"],
        model_name=config["model"],
    )
    print(llm)

    conversation = ConversationChain(llm=llm, memory=memory)
    msg = conversation(prompt)["response"]
    return msg


def on_change_chat(*args, **kwargs):
    output = generate_chat_response(st.session_state.input)

    st.session_state.chat.extend(
        [
            {"type": "user", "content": st.session_state.input},
            {"type": "bot", "content": output},
        ]
    )


def get_chat_input_text():
    input_text = st.text_input(
        "You: ",
        placeholder="Ask here any question related to the uploaded file",
        key="input",
        on_change=on_change_chat,
    )
    return input_text


def memory_buffer():
    memory = ConversationBufferMemory(return_messages=True)

    for i, msg in list(enumerate(st.session_state.chat)):
        if msg["type"] == "user":
            memory.chat_memory.add_user_message(msg["content"])
        else:
            memory.chat_memory.add_ai_message(msg["content"])
    return memory


config = read_url_param_values()

DEFAULT_CONFIG = {
    "temperature": config["temperature"],
    "max_tokens": config["max_tokens"],
    "top_p": config["top_p"],
}

setup_chat_session()
get_chat_input_text()

if st.session_state.chat:
    for i, msg in reversed(list(enumerate(st.session_state.chat))):
        message(msg["content"], is_user=msg["type"] == "user", key=str(i))
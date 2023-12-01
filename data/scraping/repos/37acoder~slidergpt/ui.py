# First
import streamlit as st
from azure import AzureGPT
import config
from langchain.schema import HumanMessage, SystemMessage, AIMessage

import auth as _

with st.sidebar:
    deployment_name = st.selectbox("deployment_name", ["gpt35-16k", "gpt4-32k"])

print(config.openai_api_version)
chater = AzureGPT(
    deployment_name or config.deployment_name,
    config.openai_api_type,
    config.openai_api_key,
    config.openai_api_base,
    config.openai_api_version,
)


st.title("💬Personal ChatGPT UI")
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hello, I'm a chatbot. How can I help you?"}
    ]


def transform_dict_to_msg(data):
    if data["role"] == "user":
        return HumanMessage(content=data["content"])
    elif data["role"] == "assistant":
        return AIMessage(content=data["content"])
    elif data["role"] == "system":
        return SystemMessage(content=data["content"])
    return None


def transform_sesssion_state_to_messages(messages):
    return [transform_dict_to_msg(msg) for msg in messages]


for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    status = st.status("Generating...")
    response = chater.chat.stream(
        transform_sesssion_state_to_messages(st.session_state.messages)
    )
    result = ""
    with st.chat_message("assistant"):
        text_element = st.markdown(result)
        for chunk in response:
            result += chunk.content
            text_element.markdown(result)
    status.update(label="complete", state="complete", expanded=False)
    st.session_state.messages.append({"role": "assistant", "content": result})


def clear_session():
    st.session_state.messages = st.session_state.messages[:1]


if len(st.session_state.messages) > 1:
    st.button("clear", on_click=clear_session)

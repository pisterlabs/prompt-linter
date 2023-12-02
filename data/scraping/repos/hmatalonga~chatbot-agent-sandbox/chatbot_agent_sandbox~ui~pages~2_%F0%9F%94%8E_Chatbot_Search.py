import os
import streamlit as st

from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import AzureChatOpenAI
from langchain.tools import DuckDuckGoSearchRun

_ = load_dotenv()

st.set_page_config(page_title="Chatbot Search", page_icon="🔎")

with st.sidebar:
    openai_api_type = st.text_input(
        "OpenAI API Type", value=os.environ["OPENAI_API_TYPE"], key="chatbot_api_type"
    )
    openai_api_base = st.text_input(
        "OpenAI API Base", value=os.environ["OPENAI_API_BASE"], key="chatbot_api_base"
    )
    openai_api_version = st.text_input(
        "OpenAI API Version",
        value=os.environ["OPENAI_API_VERSION"],
        key="chatbot_api_version",
    )
    openai_api_key = st.text_input(
        "OpenAI API Key",
        value=os.environ["OPENAI_API_KEY"],
        key="chatbot_api_key",
        type="password",
    )
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

st.title("🔎 Chatbot Search")

"""
In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
Try more LangChain 🤝 Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
"""

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Hi, I'm a chatbot who can search the web. How can I help you?",
        }
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    llm = AzureChatOpenAI(
        deployment_name=os.environ["DEPLOYMENT_NAME"],
        openai_api_key=openai_api_key,
        openai_api_base=openai_api_base,
        openai_api_version=openai_api_version,
        openai_api_type=openai_api_type,
        streaming=True,
    )
    search = DuckDuckGoSearchRun(name="Search")
    search_agent = initialize_agent(
        [search],
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
    )
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)

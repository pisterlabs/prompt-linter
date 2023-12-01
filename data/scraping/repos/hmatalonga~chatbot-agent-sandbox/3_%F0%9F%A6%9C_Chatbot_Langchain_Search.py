import os
import streamlit as st

from dotenv import load_dotenv
from langchain.agents import ConversationalChatAgent, AgentExecutor
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.tools import DuckDuckGoSearchRun

_ = load_dotenv()

st.set_page_config(page_title="LangChain: Chat with search", page_icon="🦜")
st.title("🦜 LangChain: Chat with search")

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

msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(
    chat_memory=msgs,
    return_messages=True,
    memory_key="chat_history",
    output_key="output",
)
if len(msgs.messages) == 0 or st.sidebar.button("Reset chat history"):
    msgs.clear()
    msgs.add_ai_message("How can I help you?")
    st.session_state.steps = {}

avatars = {"human": "user", "ai": "assistant"}
for idx, msg in enumerate(msgs.messages):
    with st.chat_message(avatars[msg.type]):
        # Render intermediate steps if any were saved
        for step in st.session_state.steps.get(str(idx), []):
            if step[0].tool == "_Exception":
                continue
            with st.status(
                f"**{step[0].tool}**: {step[0].tool_input}", state="complete"
            ):
                st.write(step[0].log)
                st.write(step[1])
        st.write(msg.content)

if prompt := st.chat_input(placeholder="Who won the Women's U.S. Open in 2018?"):
    st.chat_message("user").write(prompt)

    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    llm = AzureChatOpenAI(
        deployment_name=os.environ["DEPLOYMENT_NAME"],
        openai_api_key=openai_api_key,
        openai_api_base=openai_api_base,
        openai_api_version=openai_api_version,
        openai_api_type=openai_api_type,
        streaming=True,
    )
    tools = [DuckDuckGoSearchRun(name="Search")]
    chat_agent = ConversationalChatAgent.from_llm_and_tools(llm=llm, tools=tools)
    executor = AgentExecutor.from_agent_and_tools(
        agent=chat_agent,
        tools=tools,
        memory=memory,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
    )
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = executor(prompt, callbacks=[st_cb])
        st.write(response["output"])
        st.session_state.steps[str(len(msgs.messages) - 1)] = response[
            "intermediate_steps"
        ]

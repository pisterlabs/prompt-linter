# pylint: disable=invalid-name
# pylint: disable=non-ascii-file-name
""" Chat with PDF data """

import os

import streamlit as st
from langchain.agents import AgentExecutor, ConversationalChatAgent
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.tools import DuckDuckGoSearchRun

st.set_page_config(page_title="LangChain: Chat with search", page_icon="🦜")
st.title("🦜 LangChain: Chat with search")


if os.environ.get("OPENAI_API_KEY") is not None:
    openai_api_key = os.getenv("OPENAI_API_KEY")
else:
    st.error("OPENAI_API_KEY environment variable not set")
    st.stop()

msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(
    chat_memory=msgs,
    return_messages=True,
    memory_key="chat_history",
    output_key="output",
)

if st.button("New chat"):
    msgs.clear()
    st.session_state.steps = {}


if len(msgs.messages) == 0:
    msgs.clear()
    st.session_state.steps = {}

avatars = {"human": "user", "ai": "assistant"}
for idx, msg in enumerate(msgs.messages):
    with st.chat_message(avatars[msg.type]):
        # Render intermediate steps if any were saved
        for step in st.session_state.steps.get(str(idx), []):
            if step[0].tool == "_Exception":
                continue
            with st.expander(f"✅ **{step[0].tool}**: {step[0].tool_input}"):
                st.write(step[0].log)
                st.write(f"**{step[1]}**")
        st.write(msg.content)

if prompt := st.chat_input(placeholder="Send a message"):
    st.chat_message("user").write(prompt)

    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        openai_api_key=openai_api_key,
        streaming=True,
    )
    tools = [DuckDuckGoSearchRun(name="Search")]
    chat_agent = ConversationalChatAgent.from_llm_and_tools(
        llm=llm, tools=tools
    )
    executor = AgentExecutor.from_agent_and_tools(
        agent=chat_agent,
        tools=tools,
        memory=memory,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
    )
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(
            st.container(), expand_new_thoughts=False
        )
        response = executor(prompt, callbacks=[st_cb])
        st.write(response["output"])
        st.session_state.steps[str(len(msgs.messages) - 1)] = response[
            "intermediate_steps"
        ]

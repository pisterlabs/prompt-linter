import streamlit as st
from langchain.agents import AgentType, initialize_agent, AgentExecutor
from langchain.callbacks.streamlit import StreamlitCallbackHandler

import main
import config


def init(tools_select):
    tool = []
    for i in main.tools:
        if i.name in tools_select:
            tool.append(i)
    global agent
    agent = AgentExecutor.from_agent_and_tools(
        agent=main.get_agent(tool), tools=tool, verbose=True, memory=main.memory
    )


language = config.language


def _main_():
    if 'user' not in st.session_state:
        st.session_state.user = []
    if 'response' not in st.session_state:
        st.session_state.response = []
    st.set_page_config(page_title="XZAITool", page_icon="logo.png")
    st.title(title)
    if tool_select := st.multiselect('Select tools', [i.name for i in main.tools],
                                     default=[i.name for i in main.tools]):
        init(tool_select)
    if st.button("清空记忆"):
        main.memory.clear()
        st.session_state.response = []
        st.session_state.user = []
    for i in range(len(st.session_state.user) if len(st.session_state.user) >= len(st.session_state.response) else len(st.session_state.response)):
        try:
            st.chat_message("user").write(st.session_state.user[i])
            st.chat_message("assistant").write(st.session_state.response[i])
        except:
            pass
    if prompt := st.chat_input():
        st.session_state.user.append(prompt)
        st.chat_message("user").write(prompt)
        with st.chat_message("assistant"):
            st_callback = StreamlitCallbackHandler(st.container())
            response = agent.run(input=prompt + "answer with language:" + language, callbacks=[st_callback])
            st.session_state.response.append(response)
            st.write(response)


if __name__ == "__main__":
    _main_()

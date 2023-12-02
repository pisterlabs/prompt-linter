from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
from langchain.llms.openai import OpenAI
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
import streamlit as st
agent_executor = create_python_agent(
    llm=OpenAI(temperature=0, max_tokens=1000,openai_api_base=st.session_state.openai_api_base,openai_api_key=st.session_state.openai_api_key),
    tool=PythonREPLTool(),
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

prompt_text = st.chat_input("请输入您的问题")
if prompt_text:
    ans = agent_executor.run(prompt_text)
    st.code(ans,language='python')
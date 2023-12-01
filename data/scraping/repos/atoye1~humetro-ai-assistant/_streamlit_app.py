# code from https://python.langchain.com/docs/integrations/callbacks/streamlit?ref=blog.langchain.dev#installation-and-setup

from ..agent_with_tools import agent_executor, agent_chain
from langchain.callbacks import StreamlitCallbackHandler
from langchain.llms.openai import OpenAI
from langchain.agents import AgentType, initialize_agent, load_tools
import streamlit as st
from langchain.tools.ddg_search.tool import DuckDuckGoSearchRun
from langchain.tools.ddg_search.tool import DuckDuckGoSearchResults
from langchain.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from dotenv import load_dotenv, find_dotenv
import sys
sys.path.append('../..')

_ = load_dotenv(find_dotenv())  # read local .env file


llm = OpenAI(temperature=0.9, streaming=True)
wrapper = DuckDuckGoSearchAPIWrapper(region="ko-KR", time="d", max_results=10)
tools = [DuckDuckGoSearchResults(api_wrapper=wrapper, backend='news')]
tools = load_tools(['ddg-search'])

agent = agent_chain

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(
            st.container(), collapse_completed_thoughts=False)
        response = agent_chain.run(prompt, callbacks=[st_callback])
        st.write(response)

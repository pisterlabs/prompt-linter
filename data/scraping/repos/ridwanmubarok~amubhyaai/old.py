import os
from apikey import apikey, serpapikey

import streamlit as st

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAI

os.environ['OPENAI_API_KEY'] = apikey
os.environ['SERPAPI_API_KEY'] = serpapikey

st.title('AMUBHYA AI | EXPERIMENTAL')
prompt = st.text_input('Whatever You Want !')

# LLM
llm = OpenAI(temperature=0)
tool_names = ["serpapi"]
tools = load_tools(tool_names)
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

# PROMPT
if prompt:
    response = agent.run(prompt)
    st.write(response)
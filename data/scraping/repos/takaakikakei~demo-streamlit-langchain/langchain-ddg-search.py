import os

import openai
import streamlit as st
from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks import StreamlitCallbackHandler
from langchain.llms import OpenAI

# 環境変数の読み込み
load_dotenv(".env.local")
openai.api_key = os.getenv("OPENAI_API_KEY")

# タイトルを表示する
st.title("🦜🔗 Langchain DuckDuckGo Search")

# LangChainのAgentの初期化
llm = OpenAI(streaming=True)
tools = load_tools(["ddg-search"])
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

# ユーザープロンプトの入力があれば、Agentを実行してレスポンスを表示する
if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent.run(prompt, callbacks=[st_callback])
        st.write(response)

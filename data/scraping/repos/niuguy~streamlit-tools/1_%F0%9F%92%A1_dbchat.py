from langchain.sql_database import SQLDatabase
from langchain.llms import OpenAI
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
import streamlit as st
import os
import dotenv

dotenv.load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.title("Talk to your data")

db = SQLDatabase.from_uri(DATABASE_URL)
toolkit = SQLDatabaseToolkit(
    db=db, llm=OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
)
agent_executor = create_sql_agent(
    llm=OpenAI(temperature=0, streaming=True, openai_api_key=OPENAI_API_KEY),
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent_executor.run(prompt, callbacks=[st_callback])
        st.write(response)

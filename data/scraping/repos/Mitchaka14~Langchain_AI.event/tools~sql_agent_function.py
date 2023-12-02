# sql_agent_function.py
import os
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI
from langchain.agents import AgentExecutor
from langchain.agents.agent_types import AgentType
from sqlalchemy import create_engine

import streamlit as st
from dotenv import load_dotenv

try:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

try:
    os.environ["serpapi_api_key"] = st.secrets["SERPAPI_API_KEY"]
except Exception:
    load_dotenv()
    os.environ["serpapi_api_key"] = os.getenv("SERPAPI_API_KEY")


def sql_agent_function(query: str) -> str:
    """Executes the given query on a SQL database using a SQL agent."""

    # Use SQLAlchemy's create_engine to establish a connection to the SQLite database.
    engine = create_engine("sqlite:///ClinicDb.db")
    db = SQLDatabase(engine)

    toolkit = SQLDatabaseToolkit(db=db, llm=OpenAI(temperature=0))
    agent_executor = create_sql_agent(
        llm=OpenAI(temperature=0),
        toolkit=toolkit,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )

    output = agent_executor.run(query)

    return output

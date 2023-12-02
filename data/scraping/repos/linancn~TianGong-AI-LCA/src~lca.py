import os

# import brightway2
import streamlit as st
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_experimental.tools import PythonREPLTool

os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
os.environ["XATA_API_KEY"] = st.secrets["xata_api_key"]
os.environ["XATA_DATABASE_URL"] = st.secrets["xata_db_url"]


llm = ChatOpenAI(temperature=0, model="gpt-4-1106-preview")

agent_executor = create_python_agent(
    llm=llm,
    tool=PythonREPLTool(),
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    agent_executor_kwargs={"handle_parsing_errors": True},
)

prompt = """Using Brightway2:
A new project named 'simple_lca_project' was created.
A new database called 'simple_db' was defined, containing two processes: extraction of raw materials and CO2 emissions.
A simple impact assessment method named 'simple_method' was defined, which assumes an impact of 1 impact unit per kilogram of CO2 emissions.
A functional unit, specifically the extraction of 1 kilogram of raw materials, was defined.
An LCA object was initialized, and both the inventory analysis (.lci()) and impact assessment (.lcia()) were executed.
The LCA score for the extraction of raw materials was printed.
This does not include an actual LCA database; all data are hypothetical."""

agent_executor.run(prompt)

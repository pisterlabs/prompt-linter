from langchain.llms.openai import OpenAI
from langchain.agents.agent_toolkits import create_python_agent
from langchain.agents.agent_types import AgentType
from langchain.tools.python.tool import PythonREPLTool


from langchain.chat_models import ChatOpenAI

import os 

import openai
import dotenv

dotenv.load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")

agent_executor = create_python_agent(
    llm = OpenAI(temperature=0.5, max_tokens=2000),
    tool = PythonREPLTool(),
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

agent_executor.run(
    "What is the fibonacci number of 10?"
)
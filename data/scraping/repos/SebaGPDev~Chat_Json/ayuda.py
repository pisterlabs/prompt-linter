import os
import yaml

from langchain.agents import create_json_agent, AgentExecutor
from langchain.agents.agent_toolkits import JsonToolkit
from langchain.chains import LLMChain
from langchain.llms.openai import OpenAI
from langchain.requests import TextRequestsWrapper
from langchain.tools.json.tool import JsonSpec
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI

load_dotenv()

with open("alumnos.json") as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
json_spec = JsonSpec(dict_=data, max_value_length=4000)
json_toolkit = JsonToolkit(spec=json_spec)

json_agent_executor = create_json_agent(
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
, toolkit=json_toolkit, verbose=True
)

json_agent_executor.run(
"Hazme el porcentaje de presente de todos los alumnos"
)


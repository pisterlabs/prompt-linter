from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
from langchain.llms.openai import OpenAI

OPENAI_API = "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"

agent_executor = create_python_agent(
  llm=OpenAI(temperature=0, max_tokens=1000, openai_api_key=OPENAI_API),
  tool=PythonREPLTool(),
  verbose=True
)
agent_executor.run("Find the roots (zeros) if the quadratic function 3 * x**2 + 2*x - 1")

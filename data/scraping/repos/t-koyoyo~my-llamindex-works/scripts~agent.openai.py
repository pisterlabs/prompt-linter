from llama_index.tools import BaseTool, FunctionTool
from llama_index.agent import OpenAIAgent
from llama_index.llms import OpenAI
import common


# ------------------------------
# ■ Requirements
# https://gpt-index.readthedocs.io/en/v0.8.16/examples/agent/openai_agent.html
# ------------------------------

def multiply(a: int, b: int) -> int:
  """Multiple two integers and returns the result integer"""
  return a * b

def add(a: int, b: int) -> int:
  """Add two integers and returns the result integer"""
  return a + b

multiply_tool = FunctionTool.from_defaults(fn=multiply)
add_tool = FunctionTool.from_defaults(fn=add)

llm = common.llm_openai()
agent = OpenAIAgent.from_tools([multiply_tool, add_tool], llm=llm, verbose=True)

response = agent.chat("What is (121 * 3) + 42?")
print(str(response))

# inspect sources
print(response.sources)
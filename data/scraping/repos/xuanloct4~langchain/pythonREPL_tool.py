

import environment

from langchain import LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from llms import defaultLLM as llm


from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
agent_executor = create_python_agent(
    llm=llm,
    tool=PythonREPLTool(),
    verbose=True
)

# agent_executor.run("What is the 10th fibonacci number?")

# agent_executor.run("""Understand, write a single neuron neural network in PyTorch.
# Take synthetic data for y=2x. Train for 1000 epochs and print every 100 epochs.
# Return prediction for x = 5""")
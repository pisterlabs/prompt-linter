from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.tools import ShellTool
from langchain.chat_models import ChatOpenAI
import os

shell_tool = ShellTool()

os.environ['OPENAI_API_KEY'] = "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"

llm = ChatOpenAI(temperature=0.5)

shell_tool.description = shell_tool.description + f"args{shell_tool.args}".replace(
    "{","{{"
).replace("}","}}")

agent = initialize_agent(
    [shell_tool],llm,agent = AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,verbose = True
)

agent.run(
    "Create a text file called cnn_train and inside it , add the python code that trains a basic convolutional neural network for 50 epochs"
)
from langchain.agents import AgentType
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import ShellTool

shell_tool = ShellTool()

llm = ChatOpenAI(temperature=0)

shell_tool.description += f"args {shell_tool.args}".replace("{", "{{").replace(
    "}", "}}"
)
agent = initialize_agent(
    tools=[shell_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
)

agent.run(
    """do not run the codes. """
    """just create a text file called empty.txt and inside it, """
    """add code that uses PyTorch framework """
    """for training a basic convolutional neural network for 4 epochs"""
)

from langchain.agents import Tool
from pydantic import BaseModel, Field
from langchain.tools import ShellTool

class DocsInput(BaseModel):
    question: str = Field()

shell_tool = ShellTool()

# shell_tool.description = shell_tool.description + f"args {shell_tool.args}".replace("{", "{{").replace("}", "}}")
shell_tool.description = "Run shell commands on this Linux machine.args {{'commands': {{'title': 'Command', 'description': 'A shell command in string to run. Deserialized using json.loads', 'anyOf': [{{'type': 'string'}}]}}}}"

def AShellTool():
    tools = []
    tools.append(Tool(
        name = "shell",
        func=shell_tool.run,
        coroutine=shell_tool.arun,
        description=shell_tool.description
    ))
    return tools

def shell():
    tools = []
    tools.extend(AShellTool())
    return tools
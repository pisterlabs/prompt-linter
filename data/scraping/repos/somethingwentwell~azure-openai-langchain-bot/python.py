from langchain.agents import Tool
from pydantic import BaseModel, Field
from langchain.utilities import PythonREPL

class DocsInput(BaseModel):
    question: str = Field()
python_repl = PythonREPL()

def PythonTool():
    tools = []
    tools.append(Tool(
        name="python_repl",
        func=python_repl.run,
        description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
        args_schema=DocsInput
    ))
    return tools

def python():
    tools = []
    tools.extend(PythonTool())
    return tools